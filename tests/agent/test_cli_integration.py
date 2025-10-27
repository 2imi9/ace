"""Integration tests for the ACE agent CLI."""

from __future__ import annotations

import json

from fme.agent import cli
from fme.agent.session_state import SessionState


class DummyOrchestrator:
    """Test double for :class:`AgentOrchestrator`."""

    def __init__(self) -> None:
        self.session_state = SessionState()

    def run_command(self, command: str, *args, **kwargs):  # type: ignore[override]
        self.session_state.record_event(command, args=list(args), kwargs=dict(kwargs))
        if command == "run simulation":
            result = {"status": "ok", "args": list(args), "kwargs": dict(kwargs)}
            self.session_state.last_simulation_output = result
            return result
        if command == "analyze output":
            analysis = {"summary": "analysis complete", "label": kwargs.get("label")}
            self.session_state.last_analysis = analysis
            return analysis
        if command == "ask llm":
            question = args[0] if args else ""
            answer = f"answer:{question}"
            self.session_state.append_conversation("user", question)
            self.session_state.append_conversation("assistant", answer)
            return answer
        return {"command": command}


def test_cli_launch_status_export(tmp_path, monkeypatch, capsys):
    scenario = [
        {"command": "run simulation", "args": ["demo"], "kwargs": {"steps": 2}},
        {"command": "analyze output", "kwargs": {"label": "baseline"}},
        {"command": "ask llm", "args": ["How did it go?"]},
    ]
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    state_path = tmp_path / "state.json"

    orchestrators: list[DummyOrchestrator] = []

    def fake_builder(config_path):
        orchestrator = DummyOrchestrator()
        orchestrators.append(orchestrator)
        return orchestrator

    monkeypatch.setattr(cli, "build_orchestrator", fake_builder)

    exit_code = cli.main(
        [
            "launch",
            "--scenario",
            str(scenario_path),
            "--state-file",
            str(state_path),
        ]
    )
    assert exit_code == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["results"][0]["command"] == "run simulation"
    assert state_path.exists()

    summary_code = cli.main(["status", "--state-file", str(state_path)])
    assert summary_code == 0
    summary_output = capsys.readouterr().out
    summary = json.loads(summary_output)
    assert len(summary["events"]) == 3
    assert summary["has_analysis"] is True

    report_code = cli.main(
        ["export-report", "--state-file", str(state_path), "--format", "markdown"]
    )
    assert report_code == 0
    report = capsys.readouterr().out
    assert "# ACE Agent Session Report" in report
    assert "analysis complete" in report

    # Ensure the orchestrator used during launch recorded conversation turns.
    assert orchestrators[0].session_state.conversation
