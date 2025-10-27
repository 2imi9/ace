import json
from pathlib import Path

from fme.agent import AgentConfig, AgentOrchestrator, SessionState, VerificationLayer


def test_validation_flags_banned_content_and_logs(tmp_path: Path) -> None:
    validator = VerificationLayer(
        banned_phrases={"forbidden": "Prohibited content."},
        log_dir=tmp_path,
        refusal_template="Refused: {reasons}",
    )

    result = validator.validate("normal prompt", "This response includes forbidden details.")

    assert not result.is_valid
    assert "forbidden" in "; ".join(result.issues)
    assert result.final_response.startswith("Refused: ")

    log_path = validator.log_path
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "Log should contain at least one entry."
    last_entry = json.loads(contents[-1])
    assert last_entry["is_valid"] is False
    assert "forbidden" in last_entry["issues"][0]


def test_validation_requires_citation(tmp_path: Path) -> None:
    validator = VerificationLayer(banned_phrases={}, log_dir=tmp_path, refusal_template="No cite: {reasons}")

    result = validator.validate("prompt", "Response without references")

    assert not result.is_valid
    assert any("citation" in issue for issue in result.issues)
    assert result.final_response.startswith("No cite: ")


def test_validation_cross_checks_expected_facts(tmp_path: Path) -> None:
    validator = VerificationLayer(banned_phrases={}, log_dir=tmp_path, refusal_template="Mismatch: {reasons}")

    context = {"expected_facts": ["Ocean heat content is rising"]}
    response = "Ocean cooling is happening. [citation:report]"

    result = validator.validate("prompt", response, context=context)

    assert not result.is_valid
    assert any("expected fact" in issue for issue in result.issues)
    assert result.final_response.startswith("Mismatch: ")


def test_orchestrator_uses_validator_for_valid_response(tmp_path: Path) -> None:
    validator = VerificationLayer(banned_phrases={}, log_dir=tmp_path)
    session = SessionState()
    session.last_analysis = {"claims": ["Ocean heat content is rising."]}

    def fake_llm(prompt: str, config: AgentConfig) -> str:
        return "Ocean heat content is rising. [citation:ace]"

    orchestrator = AgentOrchestrator(AgentConfig(), session_state=session, llm_client=fake_llm, validator=validator)

    response = orchestrator.handle_ask_llm("Explain the trend")

    assert response == "Ocean heat content is rising. [citation:ace]"
    assert session.conversation[-1]["content"] == response


def test_orchestrator_refuses_on_invalid_response(tmp_path: Path) -> None:
    validator = VerificationLayer(banned_phrases={}, log_dir=tmp_path, refusal_template="Unable: {reasons}")
    session = SessionState()
    session.last_analysis = {"claims": ["Ocean heat content is rising."]}

    def fake_llm(prompt: str, config: AgentConfig) -> str:
        return "Ocean cooling dominates. [citation:ace]"

    orchestrator = AgentOrchestrator(AgentConfig(), session_state=session, llm_client=fake_llm, validator=validator)

    response = orchestrator.handle_ask_llm("Explain the trend")

    assert response.startswith("Unable: ")
    assert session.conversation[-1]["content"].startswith("Unable: ")
    assert session.conversation[-2]["content"] == "Explain the trend"

