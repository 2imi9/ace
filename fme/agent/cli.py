"""Command line interface for orchestrating ACE agent workflows."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from .orchestrator import AgentConfig, AgentOrchestrator
from .session_state import SessionState


ScenarioStep = Dict[str, Any]


def build_orchestrator(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env_prefix: str = "ACE_AGENT_",
    overrides: Mapping[str, Any] | None = None,
) -> AgentOrchestrator:
    """Create an :class:`AgentOrchestrator` instance from common inputs."""

    config = AgentConfig.from_sources(
        yaml_path=config_path,
        env_prefix=env_prefix,
        overrides=overrides,
    )
    return AgentOrchestrator(config=config)


def _coerce_json(value: str) -> Any:
    """Attempt to parse *value* as JSON, falling back to the original string."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_kwargs(pairs: Iterable[str]) -> Dict[str, Any]:
    """Parse ``key=value`` assignments into a dictionary."""

    parsed: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                "Keyword arguments must be supplied in the form 'key=value'."
            )
        key, raw_value = item.split("=", 1)
        parsed[key] = _coerce_json(raw_value)
    return parsed


def _load_scenario(path: Path) -> List[ScenarioStep]:
    """Load a scenario definition from ``path``."""

    data: Any
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML scenarios.")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Sequence):
        raise TypeError("Scenario definitions must be a list of command objects.")
    steps: List[ScenarioStep] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, MutableMapping):
            raise TypeError(
                f"Scenario step {index} must be a mapping with a 'command' key."
            )
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError(f"Scenario step {index} must include a non-empty command.")
        raw_args = entry.get("args", [])
        if raw_args is None:
            raw_args = []
        if isinstance(raw_args, (str, bytes)):
            raise TypeError(f"Scenario step {index} args must be a sequence, not string.")
        raw_kwargs = entry.get("kwargs", {})
        if raw_kwargs is None:
            raw_kwargs = {}
        if not isinstance(raw_kwargs, Mapping):
            raise TypeError(f"Scenario step {index} kwargs must be a mapping.")
        step = {
            "command": command,
            "args": list(raw_args),
            "kwargs": dict(raw_kwargs),
        }
        steps.append(step)
    return steps


def _run_scenario(orchestrator: AgentOrchestrator, steps: Sequence[ScenarioStep]) -> List[Any]:
    """Execute ``steps`` sequentially and return their results."""

    results: List[Any] = []
    for step in steps:
        command = step["command"]
        args = step.get("args", [])
        kwargs = step.get("kwargs", {})
        if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
            raise TypeError("Scenario 'args' must be a sequence.")
        if not isinstance(kwargs, Mapping):
            raise TypeError("Scenario 'kwargs' must be a mapping.")
        outcome = orchestrator.run_command(command, *list(args), **dict(kwargs))
        results.append({"command": command, "result": outcome})
    return results


def _serialize_session_state(state: SessionState) -> Dict[str, Any]:
    """Return a JSON-serialisable representation of the session state."""

    serialised = asdict(state)
    # ``asdict`` will recursively convert dataclasses, but not arbitrary objects.
    # Fall back to ``repr`` for values that cannot be encoded by ``json``.
    if serialised.get("last_simulation_output") is not None:
        serialised["last_simulation_output"] = _ensure_jsonable(
            serialised["last_simulation_output"]
        )
    if serialised.get("last_analysis") is not None:
        serialised["last_analysis"] = _ensure_jsonable(serialised["last_analysis"])
    return serialised


def _ensure_jsonable(value: Any) -> Any:
    """Ensure ``value`` can be serialised to JSON by coercing unknown objects."""

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {k: _ensure_jsonable(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_ensure_jsonable(item) for item in value]
        return repr(value)


def generate_report(state: SessionState, *, format: str = "json") -> str:
    """Generate a textual report of the session state."""

    format_lower = format.lower()
    payload = _serialize_session_state(state)
    if format_lower == "json":
        return json.dumps(payload, indent=2, sort_keys=True)
    if format_lower in {"md", "markdown"}:
        lines = ["# ACE Agent Session Report", ""]
        if payload["events"]:
            lines.append("## Events")
            for event in payload["events"]:
                command = event.get("command", "unknown")
                details = event.get("payload", {})
                lines.append(f"- **{command}**: {json.dumps(details)}")
            lines.append("")
        if payload["conversation"]:
            lines.append("## Conversation")
            for turn in payload["conversation"]:
                role = turn.get("role", "unknown").title()
                content = turn.get("content", "")
                lines.append(f"- **{role}**: {content}")
            lines.append("")
        if payload.get("last_analysis"):
            lines.append("## Last Analysis")
            lines.append("```json")
            lines.append(json.dumps(payload["last_analysis"], indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
        return "\n".join(lines).strip() + "\n"
    raise ValueError("Unsupported report format. Choose 'json' or 'markdown'.")


def _handle_launch(args: argparse.Namespace) -> int:
    orchestrator = build_orchestrator(args.config)
    if args.scenario is not None:
        scenario_path = Path(args.scenario)
        steps = _load_scenario(scenario_path)
    else:
        if not args.command:
            raise SystemExit("Either --scenario or --command must be provided.")
        kwargs = _parse_kwargs(args.kwarg or [])
        steps = [
            {
                "command": args.command,
                "args": [
                    _coerce_json(value) for value in (args.arg or [])
                ],
                "kwargs": kwargs,
            }
        ]
    results = _run_scenario(orchestrator, steps)
    payload = {
        "results": _ensure_jsonable(results),
        "session": _serialize_session_state(orchestrator.session_state),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.state_file:
        Path(args.state_file).write_text(
            json.dumps(payload["session"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    if not args.state_file:
        raise SystemExit("--state-file is required to inspect run status.")
    state_path = Path(args.state_file)
    session_data = json.loads(state_path.read_text(encoding="utf-8"))
    summary = {
        "events": session_data.get("events", []),
        "conversation_length": len(session_data.get("conversation", [])),
        "has_analysis": bool(session_data.get("last_analysis")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    if not args.state_file:
        raise SystemExit("--state-file is required to export reports.")
    state_path = Path(args.state_file)
    session_state = SessionState(**json.loads(state_path.read_text(encoding="utf-8")))
    report = generate_report(session_state, format=args.format)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Return the root argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        prog="ace-agent",
        description="Utilities for launching ACE agent scenarios and exporting results.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch_parser = subparsers.add_parser(
        "launch", help="Execute a scenario definition or individual command."
    )
    launch_parser.add_argument("--config", help="Path to the agent configuration file.")
    launch_parser.add_argument(
        "--scenario",
        help="Path to a JSON/YAML file describing a sequence of commands.",
    )
    launch_parser.add_argument("--command", help="Single command to execute.")
    launch_parser.add_argument(
        "--arg",
        action="append",
        help="Positional arguments for --command. May be supplied multiple times.",
    )
    launch_parser.add_argument(
        "--kwarg",
        action="append",
        help="Keyword arguments for --command in key=value form.",
    )
    launch_parser.add_argument(
        "--state-file",
        help="Optional path for persisting session state as JSON.",
    )
    launch_parser.set_defaults(func=_handle_launch)

    status_parser = subparsers.add_parser(
        "status", help="Show a summary of a previously executed run."
    )
    status_parser.add_argument(
        "--state-file",
        required=True,
        help="Path to the JSON file created during a launch run.",
    )
    status_parser.set_defaults(func=_handle_status)

    export_parser = subparsers.add_parser(
        "export-report", help="Export a detailed session report."
    )
    export_parser.add_argument(
        "--state-file",
        required=True,
        help="Path to the JSON file created during a launch run.",
    )
    export_parser.add_argument(
        "--format",
        default="json",
        choices=["json", "markdown"],
        help="Report format to generate.",
    )
    export_parser.add_argument(
        "--output",
        help="Optional destination file for the generated report.",
    )
    export_parser.set_defaults(func=_handle_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point."""

    parser = create_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
