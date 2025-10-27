# ACE Agent Operations Overview

This guide explains how to run ACE agent workflows with the new command line
interface and optional FastAPI service. It covers authentication, basic usage
examples, and environment setup for local development.

## Environment Setup

1. Create and activate a Python environment with the project dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # includes FastAPI and test utilities
   ```

2. Export credentials for your language model provider. The CLI and API read the
   key from the ``ACE_AGENT_LLM_API_KEY`` environment variable by default:

   ```bash
   export ACE_AGENT_LLM_API_KEY="sk-your-provider-key"
   ```

3. (Optional) Create a YAML configuration file describing the model and ACE
   components to use. See ``fme/agent/orchestrator.py`` for supported options.

## Command Line Interface

The CLI is exposed through ``python -m fme.agent.cli`` or the ``ace-agent``
entry point. Run ``--help`` for the full command reference.

### Launching a Scenario

Provide a JSON or YAML file containing a list of commands to execute. Each entry
should specify the command name plus optional ``args`` and ``kwargs``:

```json
[
  {"command": "run simulation", "args": ["demo"], "kwargs": {"steps": 5}},
  {"command": "analyze output", "kwargs": {"label": "baseline"}}
]
```

Launch the scenario and persist session state to disk:

```bash
ace-agent launch --config configs/agent.yaml \
    --scenario scenarios/baseline.json \
    --state-file runs/baseline-session.json
```

The command prints a JSON summary containing command results and the current
session snapshot. The optional ``--command``/``--arg``/``--kwarg`` flags allow a
single ad-hoc command without creating a scenario file.

### Monitoring Runs

Use the ``status`` sub-command to inspect a saved session file:

```bash
ace-agent status --state-file runs/baseline-session.json
```

The output lists executed events, the number of conversation turns, and whether
an ACE analysis has been recorded.

### Exporting Reports

Generate a JSON or Markdown report from a stored session:

```bash
ace-agent export-report --state-file runs/baseline-session.json --format markdown \
    --output runs/baseline-report.md
```

## FastAPI Service

The optional service exposes similar functionality through HTTP endpoints. To
run it locally, install the development requirements and start the server with
``uvicorn``:

```bash
uvicorn fme.agent.server:create_app --factory
```

### Endpoints

* ``POST /questions`` – Submit a natural language question for the agent. The
  response contains a ``job_id`` for tracking progress.
* ``GET /jobs/{job_id}`` – Poll job status until it reports ``completed``.
* ``GET /jobs/{job_id}/report`` – Retrieve the session report as JSON (default)
  or Markdown (``?format=markdown``).

Include the ``ACE_AGENT_LLM_API_KEY`` environment variable (or configure
``AgentConfig`` otherwise) so that the orchestrator can contact the underlying
LLM provider.

## Authentication Notes

* The orchestrator reads API keys from ``AgentConfig``. Loading from the
  environment is the simplest approach when running locally.
* When deploying the FastAPI service, set the environment variable via your
  process manager or secrets store.
* Downstream providers may require additional headers or configuration—supply
  these via ``AgentConfig.experiment_defaults`` in your YAML configuration.

## Testing and Mocking

Integration tests in ``tests/agent`` demonstrate how to monkeypatch the
``build_orchestrator`` helper so that CLI and API interactions can be exercised
without contacting external LLM providers. Follow the same approach when adding
new automated checks.
