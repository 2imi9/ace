"""Integration tests for the FastAPI service wrapper."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fme.agent import server
from fme.agent.session_state import SessionState


class DummyOrchestrator:
    def __init__(self) -> None:
        self.session_state = SessionState()

    def run_command(self, command: str, *args, **kwargs):  # type: ignore[override]
        self.session_state.record_event(command)
        if command == "ask llm":
            question = args[0]
            self.session_state.append_conversation("user", question)
            answer = f"answer:{question}"
            self.session_state.append_conversation("assistant", answer)
            self.session_state.last_analysis = {"summary": "mock"}
            return answer
        raise ValueError("Unsupported command")


def test_submit_question_and_retrieve_report():
    def fake_factory(config_path):
        return DummyOrchestrator()

    app = server.create_app(fake_factory)
    client = TestClient(app)

    response = client.post("/questions", json={"question": "How is the model?"})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    if status.json()["status"] != "completed":
        status = client.get(f"/jobs/{job_id}")
    assert status.json()["status"] == "completed"
    assert status.json()["result"].startswith("answer:")

    report_json = client.get(f"/jobs/{job_id}/report")
    assert report_json.status_code == 200
    assert report_json.json()["events"]

    report_md = client.get(f"/jobs/{job_id}/report", params={"format": "markdown"})
    assert report_md.status_code == 200
    assert "ACE Agent Session Report" in report_md.text
