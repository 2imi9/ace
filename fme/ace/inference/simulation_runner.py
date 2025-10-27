"""Utilities for running ACE simulations and collecting structured metadata."""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.data_loading.getters import get_forcing_data
from fme.ace.inference.inference import (  # noqa: F401  (re-exported via __all__)
    InferenceConfig,
    InitialConditionConfig,
    get_initial_condition,
)
from fme.ace.inference.evaluator import resolve_variable_metadata
from fme.ace.stepper import load_stepper, load_stepper_config
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.timing import GlobalTimer


@dataclasses.dataclass(frozen=True)
class SimulationSummary:
    """Metadata describing the outcome of a simulation run."""

    variables: list[str]
    time_metadata: Mapping[str, Any]
    anomaly_diagnostics: Mapping[str, float]
    n_initial_conditions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": self.variables,
            "time_metadata": dict(self.time_metadata),
            "anomaly_diagnostics": dict(self.anomaly_diagnostics),
            "n_initial_conditions": self.n_initial_conditions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationSummary":
        return cls(
            variables=list(data.get("variables", [])),
            time_metadata=dict(data.get("time_metadata", {})),
            anomaly_diagnostics=dict(data.get("anomaly_diagnostics", {})),
            n_initial_conditions=int(data.get("n_initial_conditions", 0)),
        )


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    """Return type for :class:`SimulationRunner` executions."""

    summary: SimulationSummary
    experiment_dir: str
    cached: bool
    metadata_path: Path | None = None


class SimulationRunner:
    """High-level utility for running ACE simulations.

    The runner reuses the existing data loading and inference writer
    infrastructure to execute an ACE stepper and produce structured
    metadata that can be surfaced to downstream LLM prompts. Optional
    caching avoids recomputing results for repeated queries.
    """

    _ANOMALY_KEYWORDS = ("bias", "rmse", "anomaly")

    def __init__(self, config: InferenceConfig, cache_dir: str | Path | None = None):
        self._config = config
        if cache_dir is None:
            cache_dir = Path(config.experiment_dir) / "metadata"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def config(self) -> InferenceConfig:
        return self._config

    def run(
        self,
        cache_key: str | None = None,
        use_cache: bool = True,
    ) -> SimulationResult:
        """Execute the simulation.

        Args:
            cache_key: Optional key used to memoize results. When provided,
                summaries from previous runs are loaded from disk when
                available.
            use_cache: If ``False`` the simulation is re-run even when a cached
                summary exists.
        """

        if cache_key is not None and use_cache:
            cached_summary = self._load_from_cache(cache_key)
            if cached_summary is not None:
                metadata_path = self._cache_path(cache_key)
                return SimulationResult(
                    summary=cached_summary,
                    experiment_dir=self._config.experiment_dir,
                    cached=True,
                    metadata_path=metadata_path,
                )

        summary = self._execute_simulation()

        metadata_path: Path | None = None
        if cache_key is not None and use_cache:
            metadata_path = self._write_cache(cache_key, summary)

        return SimulationResult(
            summary=summary,
            experiment_dir=self._config.experiment_dir,
            cached=False,
            metadata_path=metadata_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_simulation(self) -> SimulationSummary:
        config = self._config
        Path(config.experiment_dir).mkdir(parents=True, exist_ok=True)

        timer = GlobalTimer.get_instance()
        timer.start_outer("simulation")

        with timer.context("initialization"):
            config.configure_logging(log_filename="simulation_runner.log")
            env_vars = logging_utils.retrieve_env_vars()
            beaker_url = logging_utils.log_beaker_url()
            config.configure_wandb(env_vars=env_vars, notes=beaker_url)

            if fme.using_gpu():
                torch.backends.cudnn.benchmark = True

            logging_utils.log_versions()
            logging.info(f"Current device is {fme.get_device()}")

            stepper_config = load_stepper_config(
                config.checkpoint_path, config.stepper_override
            )
            data_requirements = stepper_config.get_forcing_window_data_requirements(
                n_forward_steps=config.forward_steps_in_memory
            )

            logging.info("Loading initial condition data")
            initial_condition = get_initial_condition(
                config.initial_condition.get_dataset(),
                stepper_config.prognostic_names,
                config.labels,
            )

            stepper = load_stepper(
                config.checkpoint_path, config.stepper_override
            )
            stepper.set_eval()

            logging.info("Initializing forcing data loader")
            forcing_data = get_forcing_data(
                config=config.forcing_loader,
                total_forward_steps=config.n_forward_steps,
                window_requirements=data_requirements,
                initial_condition=initial_condition,
                surface_temperature_name=stepper.surface_temperature_name,
                ocean_fraction_name=stepper.ocean_fraction_name,
                label_override=config.labels,
            )

            if not config.allow_incompatible_dataset:
                stepper.training_dataset_info.assert_compatible_with(
                    forcing_data.dataset_info
                )

            variable_metadata = resolve_variable_metadata(
                dataset_metadata=forcing_data.variable_metadata,
                stepper_metadata=stepper.training_variable_metadata,
                stepper_all_names=stepper_config.all_names,
            )

            dataset_info = forcing_data.dataset_info.update_variable_metadata(
                variable_metadata
            )

            aggregator = config.aggregator.build(
                dataset_info=dataset_info,
                n_timesteps=config.n_forward_steps + stepper.n_ic_timesteps,
                output_dir=config.experiment_dir,
            )

            writer = config.get_data_writer(
                n_initial_conditions=forcing_data.n_initial_conditions,
                timestep=forcing_data.timestep,
                coords=forcing_data.coords,
                variable_metadata=variable_metadata,
            )

        logging.info("Starting simulation")

        with torch.no_grad():
            record_logs = get_record_to_wandb(label="inference")
            run_inference(
                predict=stepper.predict_paired,
                data=forcing_data,
                writer=writer,
                aggregator=aggregator,
                record_logs=record_logs,
            )

        with timer.context("finalization"):
            writer.finalize()
            aggregator.flush_diagnostics()

        summary_logs = aggregator.get_summary_logs()
        summary = self._build_summary(
            forcing_data=forcing_data,
            summary_logs=summary_logs,
            variable_metadata=variable_metadata,
        )

        timer.stop_outer("simulation")
        record_logs([summary_logs])

        return summary

    def _build_summary(
        self,
        forcing_data,
        summary_logs: Mapping[str, Any],
        variable_metadata: Mapping[str, Any],
    ) -> SimulationSummary:
        timestep = forcing_data.timestep
        timestep_hours = timestep.total_seconds() / 3600
        n_forward_steps = self._config.n_forward_steps

        initial_times = [str(time) for time in forcing_data.initial_time.values]
        final_times: list[str] = []
        for start in forcing_data.initial_time.values:
            if isinstance(start, (datetime.datetime, datetime.date)):
                end = start + n_forward_steps * timestep
            else:
                # cftime datetime supports timedelta arithmetic
                end = start + n_forward_steps * timestep
            final_times.append(str(end))

        time_metadata = {
            "initial_times": initial_times,
            "final_times": final_times,
            "n_forward_steps": n_forward_steps,
            "timestep_hours": timestep_hours,
        }

        anomalies = self._extract_anomalies(summary_logs)

        return SimulationSummary(
            variables=sorted(variable_metadata.keys()),
            time_metadata=time_metadata,
            anomaly_diagnostics=anomalies,
            n_initial_conditions=forcing_data.n_initial_conditions,
        )

    def _extract_anomalies(
        self, summary_logs: Mapping[str, Any]
    ) -> dict[str, float]:
        anomalies: dict[str, float] = {}
        for key, value in summary_logs.items():
            if not any(token in key for token in self._ANOMALY_KEYWORDS):
                continue
            serialized = self._serialize_scalar(value)
            if serialized is not None:
                anomalies[key] = serialized
        return anomalies

    @staticmethod
    def _serialize_scalar(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, np.generic):
            return float(value.item())
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return float(tensor.item())
            return None
        if hasattr(value, "item"):
            try:
                scalar = value.item()
            except Exception:  # pragma: no cover - defensive
                return None
            if isinstance(scalar, (int, float, np.generic)):
                return float(scalar)
        return None

    def _cache_path(self, cache_key: str) -> Path:
        safe_key = cache_key.replace(os.sep, "_") if os.sep in cache_key else cache_key
        return self._cache_dir / f"{safe_key}.json"

    def _load_from_cache(self, cache_key: str) -> SimulationSummary | None:
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return SimulationSummary.from_dict(data)

    def _write_cache(
        self, cache_key: str, summary: SimulationSummary
    ) -> Path:
        cache_path = self._cache_path(cache_key)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)
        return cache_path


__all__ = [
    "SimulationRunner",
    "SimulationSummary",
    "SimulationResult",
    "InferenceConfig",
    "InitialConditionConfig",
]
