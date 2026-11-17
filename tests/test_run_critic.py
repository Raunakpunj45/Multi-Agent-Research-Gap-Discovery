import hashlib
import json
from pathlib import Path
import sys

import pytest

import experiments.run_critic as runner
from src.agents import (
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    GapCriticReview,
    GapOriginAssessment,
)


def review(status=CriticReviewStatus.INCOMPLETE, decision=CriticDecision.REVISE):
    return GapCriticReview(
        gap_id="GAP-001", decision=decision, review_status=status,
        origin_assessment=GapOriginAssessment.UNCLEAR,
        origin_rationale="The review could not be completed.",
        reasoning="Assessment unavailable.",
    )


def test_incomplete_result_is_atomically_saved_then_runner_exits_nonzero(
    tmp_path, monkeypatch, capsys,
):
    comparison = tmp_path / "comparison.json"
    gaps = tmp_path / "gaps.json"
    validation = tmp_path / "validation.json"
    output = tmp_path / "critic_review.json"
    for path, payload in ((comparison, b'{"comparison":1}'),
                          (gaps, b'{"gaps":1}'),
                          (validation, b'{"validation":1}')):
        path.write_bytes(payload)

    prior = CriticResult(
        review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[review(CriticReviewStatus.COMPLETE, CriticDecision.ACCEPT)],
    )
    prior_text = prior.model_dump_json(indent=2)
    output.write_text(prior_text, encoding="utf-8")
    expected_fingerprints = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("comparison", comparison), ("gap_detection", gaps),
                           ("evidence_validation", validation))
    }

    class StubAgent:
        fingerprints = None

        def critique(self, _comparison, _gaps, _validation,
                     input_fingerprints=None, previous_result=None):
            self.fingerprints = input_fingerprints
            return CriticResult(
                review_status=CriticReviewStatus.INCOMPLETE,
                gap_reviews=[review()],
                errors=["GAP-001: Critic assessment failed: API quota exhausted (HTTP 429)."],
                input_fingerprints=input_fingerprints or {},
            )

        @staticmethod
        def preserve_matching_successful_reviews(result, _previous):
            return result

    agent = StubAgent()
    monkeypatch.setattr(runner, "CriticAgent", lambda: agent)
    monkeypatch.setattr(runner, "load_optional", lambda path, _model: (
        object(), None, hashlib.sha256(path.read_bytes()).hexdigest()
    ))
    monkeypatch.setattr(sys, "argv", [
        "run_critic.py", "--comparison", str(comparison), "--gaps", str(gaps),
        "--validation", str(validation), "--output", str(output),
    ])

    original_replace = runner.os.replace
    replace_observations = []

    def observe_replace(source, destination):
        source, destination = Path(source), Path(destination)
        replace_observations.append((source, destination))
        assert source.exists()
        assert output.read_text(encoding="utf-8") == prior_text
        # The temporary file is complete and parseable before replacement.
        json.loads(source.read_text(encoding="utf-8"))
        original_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", observe_replace)
    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 1
    assert agent.fingerprints == expected_fingerprints
    saved = CriticResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.review_status is CriticReviewStatus.INCOMPLETE
    assert saved.input_fingerprints == expected_fingerprints
    assert len(replace_observations) == 1
    assert not list(tmp_path.glob(".critic_review.json.*.tmp"))
    assert "Critic run incomplete" in capsys.readouterr().err


def test_runner_exits_nonzero_when_only_a_gap_review_is_incomplete(tmp_path, monkeypatch):
    class StubAgent:
        def critique(self, *_args, **_kwargs):
            return CriticResult(
                review_status=CriticReviewStatus.COMPLETE,
                gap_reviews=[review()],
            )

        @staticmethod
        def preserve_matching_successful_reviews(result, _previous):
            return result

    output = tmp_path / "critic_review.json"
    monkeypatch.setattr(runner, "CriticAgent", StubAgent)
    monkeypatch.setattr(runner, "load_optional", lambda *_args: (object(), None, None))
    # The stub inputs carry no provenance; this test isolates gap-level
    # incompleteness, so lineage checks are neutralized here (covered elsewhere).
    monkeypatch.setattr(runner, "require_recorded_fingerprint", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["run_critic.py", "--output", str(output)])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 1
    saved = CriticResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.review_status is CriticReviewStatus.COMPLETE
    assert saved.gap_reviews[0].review_status is CriticReviewStatus.INCOMPLETE


def test_input_lineage_errors_force_incomplete_status_even_if_agent_reports_complete(
    tmp_path, monkeypatch,
):
    class StubAgent:
        def critique(self, *_args, **_kwargs):
            complete = review().model_copy(update={"review_status": CriticReviewStatus.COMPLETE})
            return CriticResult(review_status=CriticReviewStatus.COMPLETE, gap_reviews=[complete])

    output = tmp_path / "critic_review.json"
    monkeypatch.setattr(runner, "CriticAgent", StubAgent)
    monkeypatch.setattr(runner, "load_optional", lambda *_args: (object(), None, None))
    monkeypatch.setattr(sys, "argv", ["run_critic.py", "--output", str(output)])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 1
    saved = CriticResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.review_status is CriticReviewStatus.INCOMPLETE
    assert any("missing provenance" in error for error in saved.errors)


def test_rerun_preserves_previous_critic_output_in_archive(tmp_path, monkeypatch):
    class StubAgent:
        def critique(self, *_args, **_kwargs):
            return CriticResult(review_status=CriticReviewStatus.COMPLETE, gap_reviews=[review()])

    output = tmp_path / "critic_review.json"
    previous_text = '{"previous": "complete critic review"}\n'
    output.write_text(previous_text, encoding="utf-8")
    monkeypatch.setattr(runner, "CriticAgent", StubAgent)
    monkeypatch.setattr(runner, "load_optional", lambda *_args: (object(), None, None))
    monkeypatch.setattr(runner, "require_recorded_fingerprint", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["run_critic.py", "--output", str(output)])

    with pytest.raises(SystemExit):
        runner.main()

    archived = list((tmp_path / "archive").glob("critic_review-*.json"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == previous_text
    assert output.read_text(encoding="utf-8") != previous_text
