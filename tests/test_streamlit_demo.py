"""Offline tests for the Streamlit demo and its workflow integration.

The UI (streamlit_app.py) and its service layer (src/workflow/demo.py) run the
same LangGraph workflow as the CLI. Here every agent is a fake (no network, no
model), runs go to temporary directories, and the real Streamlit script is
driven headlessly with streamlit.testing.v1.AppTest.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import src.workflow.langgraph_app as workflow_module
from src.agents.critic_agent import CriticAgent, _LLMGapVerdict
from src.agents.evidence_validation_agent import (
    EvidenceValidationStatus as EV,
    GapSupportStatus,
    GapValidationStatus,
)
from src.processing.pdf_downloader import PDFDownloader, PDFDownloadError
from src.retrieval.models import PaperRecord
from src.workflow import demo
from src.workflow.checkpoint import RunCheckpoint
from tests import test_langgraph_workflow as wf

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ("Multi-agent LLM systems for scientific literature review and automated "
         "research gap discovery")


def candidates(relevant=14, irrelevant=3):
    """Ranked candidates: paper-1.. are relevant (>= 0.75), extra-* are not."""
    papers, scores = [], []
    for n in range(1, relevant + 1):
        papers.append(PaperRecord(title=f"Relevant paper {n}", authors=[], year=2026, doi=None,
                                  abstract=None, source="test", source_id=f"paper-{n}",
                                  landing_url=None, pdf_url=f"https://example.org/{n}.pdf"))
        scores.append(round(0.9 - n * 0.005, 4))
    for n in range(1, irrelevant + 1):
        papers.append(PaperRecord(title=f"Unrelated paper {n}", authors=[], year=2026, doi=None,
                                  abstract=None, source="test", source_id=f"extra-{n}",
                                  landing_url=None, pdf_url=f"https://example.org/x{n}.pdf"))
        scores.append(0.5)
    return papers, scores


def fake_agents(no_pdf=(), analysis_fails=(), **candidate_kwargs):
    papers, scores = candidates(**candidate_kwargs)
    agents = wf.workflow_agents()
    agents.retrieval.search.return_value = (wf.search_plan(), papers, scores)

    def resolve(paper, paper_id, paths):
        if paper_id in no_pdf:
            raise ValueError("No PDF path or PDF URL available.")
        return "downloaded.pdf"

    def analyze(paper, **kwargs):
        if paper.source_id in analysis_fails:
            raise ValueError(f"No evidence retrieved for paper: {paper.source_id}")
        return wf.analysis(paper.source_id)

    agents.paper_analysis._resolve_pdf_path.side_effect = resolve
    agents.paper_analysis.analyze_paper.side_effect = analyze
    return agents


def run(tmp_path, agents, **settings):
    return demo.run_demo(
        demo.DemoSettings(topic=TOPIC, **settings), agents=agents,
        runs_dir=tmp_path / "runs", processed_dir=tmp_path / "processed")


# ---------------------------------------------------------------------------
# 10-paper target and separate counts
# ---------------------------------------------------------------------------

def test_ten_papers_are_analyzed_when_ten_accessible_papers_exist(tmp_path):
    agents = fake_agents(no_pdf={"paper-3", "paper-5"}, analysis_fails={"paper-7"})
    result = run(tmp_path, agents)

    counts = result.counts
    assert result.target_papers == 10 and result.target_reached
    assert counts.analyzed == 10
    assert (counts.retrieved, counts.relevant) == (17, 14)
    assert counts.downloaded == 11                  # 10 analyzed + 1 failed after download
    assert (counts.failed, counts.failed_download, counts.failed_analysis) == (3, 2, 1)
    assert (counts.skipped_below_relevance, counts.skipped_max_papers) == (3, 1)
    assert counts.skipped == 4 and counts.skipped_quota == 0
    analyzed = [c.kwargs["paper"].source_id
                for c in agents.paper_analysis.analyze_paper.call_args_list]
    assert analyzed == [f"paper-{n}" for n in (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13)]
    assert result.status == "COMPLETE"


def test_failed_pdfs_are_recorded_with_their_step_and_replaced(tmp_path):
    result = run(tmp_path, fake_agents(no_pdf={"paper-1"}, analysis_fails={"paper-2"}))
    selection = {e["paper_id"]: e for e in result.state["paper_selection"]}
    assert selection["paper-1"]["outcome"] == "failed"
    assert selection["paper-1"]["failure_step"] == "download"
    assert selection["paper-1"]["pdf_obtained"] is False
    assert "No PDF path or PDF URL available" in selection["paper-1"]["reason"]
    assert selection["paper-2"]["failure_step"] == "analysis"
    assert selection["paper-2"]["pdf_obtained"] is True
    assert selection["paper-12"]["outcome"] == "analyzed"   # replacements were taken
    assert [e.paper_id for e in result.state["excluded_papers"]] == ["paper-1", "paper-2"]
    assert "paper-1` (download step)" in result.report_markdown
    assert "- Failed: 2 (download 1, analysis 1)" in result.report_markdown


def test_fewer_accessible_papers_than_target_is_reported_not_hidden(tmp_path):
    result = run(tmp_path, fake_agents(relevant=6, no_pdf={"paper-6"}))
    assert result.counts.analyzed == 5 and not result.target_reached
    assert ("Target of 10 analyzed papers was not reached: only 5 relevant paper(s)"
            in result.report_markdown)


def test_progress_is_reported_after_every_attempted_paper(tmp_path):
    seen = []
    agents = fake_agents(no_pdf={"paper-2"})
    demo.run_demo(demo.DemoSettings(topic=TOPIC, target_papers=3), agents=agents,
                  runs_dir=tmp_path / "runs", processed_dir=tmp_path / "processed",
                  progress=lambda stage, state: seen.append(demo.describe_progress(stage, state)))
    assert seen[0] == "Retrieved and ranked candidate papers: 17 candidates."
    partial = [line for line in seen if line.startswith("Downloading and analyzing")]
    assert partial[-1] == ("Downloading and analyzing papers: 3/3 analyzed, 1 failed "
                           "(download 1, analysis 0).")
    assert len(partial) == 4                     # 3 successes + 1 failure


def test_settings_are_bounded():
    for bad in (dict(target_papers=16), dict(target_papers=1), dict(minimum_papers=1),
                dict(target_papers=3, minimum_papers=4), dict(max_results_per_source=11),
                dict(topic="  ")):
        with pytest.raises(ValueError):
            demo.DemoSettings(**{"topic": TOPIC, **bad}).validate()


# ---------------------------------------------------------------------------
# Fresh-run isolation
# ---------------------------------------------------------------------------

def test_each_demo_run_uses_a_new_directory_and_never_resumes(tmp_path):
    runs = tmp_path / "runs"
    # An earlier CLI-style run of the same topic and settings, with saved state.
    from src.workflow.checkpoint import default_run_dir, run_settings
    from src.workflow.runner import initial_state

    config = demo.DemoSettings(topic=TOPIC).config()
    old_dir = default_run_dir(run_settings(initial_state(config)), runs)
    RunCheckpoint(old_dir).ensure_manifest(run_settings(initial_state(config)))
    RunCheckpoint(old_dir).save({**initial_state(config), "workflow_status": "STOPPED"}, "critic")
    before = {p: p.read_bytes() for p in old_dir.rglob("*") if p.is_file()}

    first = run(tmp_path, fake_agents())
    second = run(tmp_path, fake_agents())

    assert first.run_dir != second.run_dir != old_dir
    assert first.output_path != second.output_path
    for result in (first, second):
        assert result.run_dir.name.startswith("ui-multi-agent-llm-systems-for-scientific")
        assert not any(e.get("action") == "auto_resume" for e in result.state.get("run_lineage", []))
        assert result.output_path.exists() and (result.run_dir / "report.md").exists()
    assert {p: p.read_bytes() for p in old_dir.rglob("*") if p.is_file()} == before


def test_new_run_paths_skip_existing_names(tmp_path):
    stamp = __import__("datetime").datetime(2026, 11, 19, tzinfo=__import__("datetime").timezone.utc)
    taken = demo.new_run_paths(TOPIC, runs_dir=tmp_path, processed_dir=tmp_path, now=stamp,
                               token="aaaaaa")
    taken.run_dir.mkdir()
    again = demo.new_run_paths(TOPIC, runs_dir=tmp_path, processed_dir=tmp_path, now=stamp,
                               token="aaaaaa")
    assert again.run_dir != taken.run_dir and not again.run_dir.exists()


def test_output_is_never_overwritten(tmp_path, monkeypatch):
    fixed = demo.RunPaths(run_dir=tmp_path / "runs" / "ui-x", output_path=tmp_path / "out.json")
    fixed.output_path.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(demo, "new_run_paths", lambda *a, **k: fixed)
    with pytest.raises(FileExistsError):
        run(tmp_path, fake_agents())
    assert fixed.output_path.read_text(encoding="utf-8") == "previous"


# ---------------------------------------------------------------------------
# Evidence-grounded rejection (as in the fresh live run's GAP-001)
# ---------------------------------------------------------------------------

def test_partially_supported_gap_is_rejected_and_gets_no_questions(tmp_path):
    agents = fake_agents()
    validation = wf.validation()
    gap = validation.gap_validations[0]
    gap.status = GapValidationStatus.PARTIALLY_SUPPORTED
    gap.joint_assessment.status = GapSupportStatus.PARTIALLY_SUPPORTED
    gap.evidence_validations[1].status = EV.PARTIALLY_SUPPORTED   # second paper's chunk
    agents.evidence_validation.validate.return_value = validation
    model_calls = []

    def accepting_model(**kwargs):
        model_calls.append(kwargs)
        return SimpleNamespace(parsed=_LLMGapVerdict(
            gap_id="GAP-001", decision="ACCEPT", origin_assessment="INFERRED",
            origin_rationale="r", reasoning="would accept", evidence_labels=["P1-E1"]))

    agents.critic = CriticAgent(client=SimpleNamespace(
        models=SimpleNamespace(generate_content=accepting_model)), max_retries=1)
    result = run(tmp_path, agents)

    assert model_calls == []                                    # deterministic REJECT
    assert result.status == "STOPPED"
    assert result.stop_reason == "Critic rejected all candidate gaps."
    assert result.gaps == [{"gap_id": "GAP-001", "title": "Candidate", "papers": 2,
                            "validation": "PARTIALLY_SUPPORTED", "critic": "REJECT",
                            "questions": 0, "hypothesis": None}]
    agents.research_questions.generate_questions.assert_not_called()


# ---------------------------------------------------------------------------
# PDF downloader
# ---------------------------------------------------------------------------

class FakeSession:
    def __init__(self, status=200, content=b"", content_type="application/pdf"):
        self.status, self.content, self.content_type = status, content, content_type

    def get(self, url, timeout, headers):
        response = requests.Response()
        response.status_code = self.status
        response._content = self.content
        response.headers["Content-Type"] = self.content_type
        response.url = url
        return response


def test_pdf_detected_by_signature_even_with_generic_content_type(tmp_path):
    downloader = PDFDownloader(output_dir=tmp_path, session=FakeSession(
        content=b"%PDF-1.7\n...", content_type="application/octet-stream"))
    path = downloader.download("https://example.org/a.pdf", "paper_a")
    assert path.read_bytes() == b"%PDF-1.7\n..."


@pytest.mark.parametrize("session, error", [
    (FakeSession(content=b"<html>Sign in</html>", content_type="application/pdf"), PDFDownloadError),
    (FakeSession(content=b""), PDFDownloadError),
    (FakeSession(status=403, content=b"Forbidden", content_type="text/html"), requests.HTTPError),
])
def test_failed_downloads_leave_no_file(tmp_path, session, error):
    downloader = PDFDownloader(output_dir=tmp_path, session=session)
    with pytest.raises(error):
        downloader.download("https://example.org/a.pdf", "paper_a")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The real Streamlit page, driven headlessly
# ---------------------------------------------------------------------------

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def loopback_only(monkeypatch):
    """Allow loopback sockets only (Streamlit's asyncio loop on Windows uses a
    127.0.0.1 socket pair); every external connection stays blocked."""
    import _socket
    import socket

    guarded, guarded_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self, address):
        if isinstance(address, tuple) and address[0] in LOOPBACK:
            return _socket.socket.connect(self, address)
        return guarded(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and address[0] in LOOPBACK:
            return _socket.socket.connect_ex(self, address)
        return guarded_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


def test_loopback_exception_still_blocks_external_connections(loopback_only):
    import socket

    blocked = "Offline test attempted a network connection"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RuntimeError, match=blocked):
            sock.connect(("8.8.8.8", 443))
    with pytest.raises(RuntimeError, match=blocked):
        socket.create_connection(("example.org", 443))


@pytest.fixture
def app(tmp_path, monkeypatch, loopback_only):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(demo, "DEFAULT_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(demo, "DEFAULT_PROCESSED_DIR", tmp_path / "processed")
    holder = {}

    def agents_factory(**_kwargs):
        holder["agents"] = fake_agents(no_pdf={"paper-3", "paper-5"}, analysis_fails={"paper-7"})
        return holder["agents"]

    monkeypatch.setattr(workflow_module, "default_workflow_agents", agents_factory)
    test = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    test.run()
    return test, holder


def button(test, label):
    return next(b for b in test.button if b.label == label)


def test_ui_run_shows_status_counts_gaps_and_report(app, tmp_path):
    test, holder = app
    assert not test.exception
    assert test.text_input[0].value == TOPIC
    assert [n.value for n in test.number_input] == [10, 2, 5]

    button(test, "Run").click().run()
    assert not test.exception
    assert holder["agents"].retrieval.search.call_count == 1   # the real workflow ran once
    assert test.subheader[0].value == "Workflow status: COMPLETE"
    metrics = {m.label: m.value for m in test.metric}
    assert metrics == {"Retrieved": "17", "Downloaded": "11", "Analyzed": "10",
                       "Failed": "3", "Skipped": "4"}
    assert any("Failed: 2 download, 1 analysis. Skipped: 3 below relevance, 1 after the "
               "target was reached, 0 because of API quota." in m.value for m in test.markdown)
    assert not test.warning                                      # target reached
    assert test.dataframe[0].value["critic"].tolist() == ["ACCEPT"]
    downloads = test.get("download_button")
    assert len(downloads) == 1
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "report.md").exists()


def test_ui_rerun_does_not_repeat_the_workflow(app):
    test, holder = app
    button(test, "Run").click().run()
    test.run()                                                    # e.g. any widget interaction
    assert holder["agents"].retrieval.search.call_count == 1
    assert test.subheader[0].value == "Workflow status: COMPLETE"


def test_ui_estimate_sends_nothing(app, tmp_path):
    test, holder = app
    button(test, "Estimate cost (no API calls)").click().run()
    assert not test.exception
    assert "agents" not in holder                                 # no agents were even built
    assert "Dry run: no request was sent." in test.code[0].value
    assert not (tmp_path / "runs").exists()                       # estimate writes nothing


def test_ui_rejects_invalid_settings_without_running(app):
    test, holder = app
    test.text_input[0].set_value("   ")
    button(test, "Run").click().run()
    assert test.error[0].value == "Enter a research topic."
    assert "agents" not in holder
