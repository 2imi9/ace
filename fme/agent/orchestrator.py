"""High level orchestration utilities for ACE driven agent workflows."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from fme.ace.aggregator import (
    InferenceEvaluatorAggregator,
    InferenceEvaluatorAggregatorConfig,
    NullAggregator,
    OneStepAggregator,
    OneStepAggregatorConfig,
    TrainAggregator,
)
from fme.ace.stepper import (
    Stepper,
    StepperOverrideConfig,
    load_stepper,
)

from .prompt_routing import PromptRouter
from .session_state import SessionState
from .validation import VerificationLayer


@dataclass
class AgentConfig:
    """Configuration settings for the ACE agent orchestrator."""

    model_name: str = "gpt-4"
    llm_api_key: Optional[str] = None
    experiment_defaults: Dict[str, Any] = field(default_factory=dict)
    aggregator_type: str = "null"
    aggregator_options: Dict[str, Any] = field(default_factory=dict)
    stepper_checkpoint: Optional[str] = None
    stepper_override: StepperOverrideConfig | Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        """Create a configuration object from a plain dictionary."""

        known_fields = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "AgentConfig":
        """Load configuration from a YAML file."""

        if yaml is None:
            raise RuntimeError("PyYAML is required to load configuration from YAML files.")
        with open(path, "r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        if not isinstance(payload, MutableMapping):
            raise TypeError("Agent configuration YAML must define a mapping at the top level.")
        return cls.from_dict(dict(payload))

    @classmethod
    def from_env(
        cls,
        prefix: str = "ACE_AGENT_",
    ) -> "AgentConfig" | None:
        """Load configuration from environment variables."""

        data: Dict[str, Any] = {}
        lookup = {
            "model_name": "MODEL_NAME",
            "llm_api_key": "LLM_API_KEY",
            "aggregator_type": "AGGREGATOR_TYPE",
            "stepper_checkpoint": "STEPPER_CHECKPOINT",
        }
        for field_name, suffix in lookup.items():
            value = os.getenv(f"{prefix}{suffix}")
            if value is not None:
                data[field_name] = value
        json_fields = {
            "experiment_defaults": "EXPERIMENT_DEFAULTS",
            "aggregator_options": "AGGREGATOR_OPTIONS",
            "stepper_override": "STEPPER_OVERRIDE",
        }
        for field_name, suffix in json_fields.items():
            value = os.getenv(f"{prefix}{suffix}")
            if value:
                try:
                    data[field_name] = json.loads(value)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Environment variable {prefix}{suffix} must contain valid JSON."
                    ) from exc
        if not data:
            return None
        return cls.from_dict(data)

    @classmethod
    def from_sources(
        cls,
        yaml_path: str | os.PathLike[str] | None = None,
        env_prefix: str = "ACE_AGENT_",
        overrides: Mapping[str, Any] | None = None,
    ) -> "AgentConfig":
        """Load configuration from YAML and environment variables with overrides."""

        merged: Dict[str, Any] = {}
        if yaml_path is not None:
            merged.update(asdict(cls.from_yaml(yaml_path)))
        env_config = cls.from_env(prefix=env_prefix)
        if env_config is not None:
            merged.update(asdict(env_config))
        if overrides:
            merged.update(dict(overrides))
        if not merged:
            return cls()
        return cls.from_dict(merged)


class AgentOrchestrator:
    """Coordinates ACE steppers, aggregators, and conversational context."""

    def __init__(
        self,
        config: AgentConfig,
        session_state: SessionState | None = None,
        llm_client: Callable[[str, AgentConfig], str] | None = None,
        validator: VerificationLayer | None = None,
    ) -> None:
        self.config = config
        self.session_state = session_state or SessionState()
        self.llm_client = llm_client
        self._stepper: Stepper | None = None
        self._aggregator: Any | None = None
        self.validator = validator
        self.router = PromptRouter(
            {
                "run simulation": self.handle_run_simulation,
                "analyze output": self.handle_analyze_output,
                "ask llm": self.handle_ask_llm,
            }
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def get_stepper(self) -> Stepper:
        """Lazily instantiate the ACE stepper declared in the configuration."""

        if self._stepper is None:
            if not self.config.stepper_checkpoint:
                raise RuntimeError("No stepper checkpoint configured.")
            override = self._load_stepper_override()
            self._stepper = load_stepper(self.config.stepper_checkpoint, override)
        return self._stepper

    def _load_stepper_override(self) -> StepperOverrideConfig | None:
        override = self.config.stepper_override
        if not override:
            return None
        if isinstance(override, StepperOverrideConfig):
            return override
        if isinstance(override, Mapping):
            allowed = {field.name for field in fields(StepperOverrideConfig)}
            kwargs = {key: override[key] for key in override if key in allowed}
            return StepperOverrideConfig(**kwargs)
        raise TypeError(
            "stepper_override must be a mapping or StepperOverrideConfig instance."
        )

    def get_aggregator(self) -> Any:
        """Lazily instantiate the configured aggregator."""

        if self._aggregator is not None:
            return self._aggregator

        aggregator_type = (self.config.aggregator_type or "null").lower()
        options = dict(self.config.aggregator_options)
        if aggregator_type == "null":
            self._aggregator = NullAggregator()
        elif aggregator_type == "train":
            self._aggregator = TrainAggregator()
        elif aggregator_type == "one_step":
            self._aggregator = self._build_one_step_aggregator(options)
        elif aggregator_type == "inference_evaluator":
            self._aggregator = self._build_inference_evaluator_aggregator(options)
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported aggregator type '{self.config.aggregator_type}'.")
        return self._aggregator

    def _build_one_step_aggregator(self, options: Dict[str, Any]) -> OneStepAggregator:
        stepper = self.get_stepper()
        dataset_info = stepper.training_dataset_info
        config_kwargs = {
            key: options[key]
            for key in ("log_snapshots", "log_mean_maps")
            if key in options
        }
        config = OneStepAggregatorConfig(**config_kwargs)
        build_kwargs = {
            key: value
            for key, value in options.items()
            if key not in config_kwargs
        }
        return config.build(dataset_info=dataset_info, **build_kwargs)

    def _build_inference_evaluator_aggregator(
        self, options: Dict[str, Any]
    ) -> InferenceEvaluatorAggregator:
        stepper = self.get_stepper()
        dataset_info = stepper.training_dataset_info
        required = {"n_timesteps", "initial_time", "normalize"}
        missing = required - options.keys()
        if missing:
            missing_values = ", ".join(sorted(missing))
            raise ValueError(
                "Inference evaluator aggregator requires options: " f"{missing_values}."
            )
        config_kwargs = {
            key: options[key]
            for key in [
                "log_histograms",
                "log_video",
                "log_extended_video",
                "log_zonal_mean_images",
                "log_seasonal_means",
                "log_global_mean_time_series",
                "log_global_mean_norm_time_series",
                "monthly_reference_data",
                "time_mean_reference_data",
            ]
            if key in options
        }
        config = InferenceEvaluatorAggregatorConfig(**config_kwargs)
        build_kwargs = {
            key: options[key]
            for key in [
                "n_timesteps",
                "initial_time",
                "normalize",
                "output_dir",
                "record_step_20",
                "channel_mean_names",
                "save_diagnostics",
            ]
            if key in options
        }
        return config.build(dataset_info=dataset_info, **build_kwargs)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def handle_run_simulation(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the configured stepper with the provided arguments."""

        stepper = self.get_stepper()
        result = stepper.predict(*args, **kwargs)
        self.session_state.last_simulation_output = result
        self.session_state.record_event("run simulation")
        return result

    def handle_analyze_output(
        self,
        output: Any,
        label: str = "analysis",
        *,
        record_args: tuple[Any, ...] | None = None,
        record_kwargs: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Analyze simulation output using the configured aggregator."""

        aggregator = self.get_aggregator()
        should_record = False
        args_to_use: tuple[Any, ...] = tuple()
        kwargs_to_use: Dict[str, Any] = {}
        if record_args is not None or record_kwargs is not None:
            args_to_use = record_args or tuple()
            kwargs_to_use = dict(record_kwargs or {})
            should_record = True
        elif isinstance(output, Mapping) and "args" in output:
            mapping = dict(output)
            args_to_use = tuple(mapping.get("args", ()))
            kwargs_to_use = dict(mapping.get("kwargs", {}))
            should_record = True
        if should_record and hasattr(aggregator, "record_batch"):
            aggregator.record_batch(*args_to_use, **kwargs_to_use)
        logs: Dict[str, Any]
        if hasattr(aggregator, "get_logs"):
            logs = dict(aggregator.get_logs(label))
        else:
            logs = {}
        self.session_state.last_analysis = logs
        self.session_state.record_event("analyze output", label=label)
        return logs

    def handle_ask_llm(self, prompt: str) -> str:
        """Route a prompt to the configured LLM backend."""

        if self.llm_client is None:
            raise RuntimeError("No LLM client configured for the orchestrator.")
        response = self.llm_client(prompt, self.config)
        final_response = response
        if self.validator is not None:
            validation_context = self._build_validation_context()
            result = self.validator.validate(prompt, response, context=validation_context)
            final_response = result.final_response
        self.session_state.append_conversation("user", prompt)
        self.session_state.append_conversation("assistant", final_response)
        self.session_state.record_event("ask llm")
        return final_response

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def run_command(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch ``command`` through the prompt router."""

        return self.router.route(command, *args, **kwargs)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _build_validation_context(self) -> Dict[str, Any]:
        """Construct context used by the verification layer."""

        context: Dict[str, Any] = {}
        analysis = self.session_state.last_analysis
        if isinstance(analysis, Mapping):
            ace_outputs: Dict[str, Any] = {}
            for key in ("claims", "facts", "summary"):
                value = analysis.get(key)
                if value is not None:
                    ace_outputs[key] = value
            if ace_outputs:
                context["ace_outputs"] = ace_outputs
            expected = analysis.get("expected_facts")
            if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
                context["expected_facts"] = list(expected)
        return context


