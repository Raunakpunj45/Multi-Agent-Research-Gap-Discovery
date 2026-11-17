import subprocess
from pathlib import Path

import pytest

from src.agents import gemini_config
from src.agents.critic_agent import CriticAgent
from src.agents.cross_paper_comparison_agent import CrossPaperComparisonAgent
from src.agents.evidence_validation_agent import EvidenceValidationAgent
from src.agents.paper_analysis_agent import PaperAnalysisAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.research_gap_detection_agent import ResearchGapDetectionAgent
from src.agents.research_question_agent import ResearchQuestionAgent
from src.evaluation.baselines import GeminiStructuredGenerator


def test_all_six_agent_groups_use_their_assigned_key_and_owned_client(monkeypatch):
    monkeypatch.setattr(gemini_config, "load_dotenv", lambda: None)
    for env_name in gemini_config.LEGACY_GROUP_KEY_ENV.values():
        monkeypatch.setenv(env_name, "old-four-key-secret")
    for group, env_name in gemini_config.GROUP_KEY_ENV.items():
        monkeypatch.setenv(env_name, f"new-{group}-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-secret")

    created = []

    class Client:
        def __init__(self, api_key):
            self.api_key = api_key
            created.append(self)

    monkeypatch.setattr(gemini_config.genai, "Client", Client)

    agents = {
        "ANALYSIS": [PaperAnalysisAgent().client, PlannerAgent().client,
                     GeminiStructuredGenerator()._client],
        "COMPARISON": [CrossPaperComparisonAgent().client],
        "GAP": [ResearchGapDetectionAgent().client],
        "VALIDATION": [EvidenceValidationAgent(vector_store=object()).client],
        "CRITIC": [CriticAgent().client],
        "QUESTION": [ResearchQuestionAgent().client],
    }
    for group, clients in agents.items():
        assert clients
        assert all(client.api_key == f"new-{group}-secret" for client in clients)

    assert len(created) == sum(map(len, agents.values()))
    assert len({id(client) for client in created}) == len(created)
    assert len({client.api_key for client in created}) == len(gemini_config.GROUP_KEY_ENV)


@pytest.mark.parametrize("group", gemini_config.GROUP_KEY_ENV)
def test_previous_four_key_is_used_before_legacy_fallback(monkeypatch, group):
    monkeypatch.setattr(gemini_config, "load_dotenv", lambda: None)
    new_name = gemini_config.GROUP_KEY_ENV[group]
    old_name = gemini_config.LEGACY_GROUP_KEY_ENV[group]
    monkeypatch.delenv(new_name, raising=False)
    monkeypatch.setenv(old_name, "old-four-key-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-secret")
    captured = []
    monkeypatch.setattr(
        gemini_config.genai, "Client",
        lambda *, api_key: captured.append(api_key) or object(),
    )

    gemini_config.create_gemini_client(group)

    assert captured == ["old-four-key-secret"]


@pytest.mark.parametrize("group", gemini_config.GROUP_KEY_ENV)
def test_legacy_key_is_final_fallback(monkeypatch, group):
    monkeypatch.setattr(gemini_config, "load_dotenv", lambda: None)
    for name in (*gemini_config.GROUP_KEY_ENV.values(),
                 *gemini_config.LEGACY_GROUP_KEY_ENV.values()):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-secret")
    captured = []
    monkeypatch.setattr(
        gemini_config.genai, "Client",
        lambda *, api_key: captured.append(api_key) or object(),
    )

    gemini_config.create_gemini_client(group)

    assert captured == ["legacy-secret"]


@pytest.mark.parametrize("group", gemini_config.GROUP_KEY_ENV)
def test_missing_key_error_lists_fallback_names_without_key_values(monkeypatch, group):
    monkeypatch.setattr(gemini_config, "load_dotenv", lambda: None)
    for name in (*gemini_config.GROUP_KEY_ENV.values(),
                 *gemini_config.LEGACY_GROUP_KEY_ENV.values(), "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    required = gemini_config.GROUP_KEY_ENV[group]
    old_name = gemini_config.LEGACY_GROUP_KEY_ENV[group]
    with pytest.raises(ValueError) as error:
        gemini_config.create_gemini_client(group)
    message = str(error.value)
    assert required in message
    assert old_name in message
    assert "GEMINI_API_KEY" in message
    assert "secret" not in message


def test_env_example_has_six_empty_key_placeholders_and_env_is_ignored():
    root = Path(__file__).resolve().parents[1]
    example = (root / ".env.example").read_text(encoding="utf-8")
    configured = {line.strip() for line in example.splitlines()}
    for name in gemini_config.GROUP_KEY_ENV.values():
        assert f"{name}=" in configured
    in_work_tree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=root,
        capture_output=True, text=True,
    ).stdout.strip() == "true"
    if in_work_tree:
        result = subprocess.run(
            ["git", "check-ignore", ".env"], cwd=root, capture_output=True,
            text=True, check=True,
        )
        assert result.stdout.strip() == ".env"
    else:
        # Exported archives have no .git directory; check the ignore rule itself.
        rules = {line.strip() for line in
                 (root / ".gitignore").read_text(encoding="utf-8").splitlines()}
        assert ".env" in rules
