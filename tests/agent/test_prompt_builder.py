"""Tests for the prompt builder templates."""

from __future__ import annotations

import pytest

from fme.agent.prompts import PromptBuilder


@pytest.fixture()
def sample_metadata() -> dict[str, object]:
    return {
        "dataset_name": "Climate Risk Dataset",
        "dataset_overview": "Historical climate indicators for multiple regions.",
        "simulation_objective": "Assess future weather risk under various emissions scenarios.",
        "assumptions": [
            "Historical trends remain informative for near-term projections.",
            "Data quality checks have removed erroneous sensor readings.",
        ],
        "time_horizon": "2030-2050",
        "scenario_parameters": {"emissions_path": "SSP2"},
    }


def test_build_prompt_renders_specialized_analysis(sample_metadata: dict[str, object]) -> None:
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        analysis_type="exploratory_data_analysis",
        simulation_metadata=sample_metadata,
        variable_descriptions=[
            {
                "name": "temperature_anomaly",
                "description": "Difference from long-term average temperature",
                "type": "float",
                "unit": "°C",
                "role": "target",
            },
            {
                "name": "co2_concentration",
                "description": "Atmospheric CO2 levels",
                "type": "float",
                "unit": "ppm",
                "role": "feature",
            },
        ],
        requested_outputs=["List high variance regions", "Summarize missing data patterns"],
        uncertainty_info={
            "sources": ["sensor drift", "model calibration"],
            "confidence_level": "medium",
            "notes": "Regional sensors vary in quality.",
        },
        user_request="Focus on coastal regions.",
        additional_context="Simulation assumes limited adaptation efforts.",
    )

    assert "Climate Risk Dataset" in prompt
    assert "Exploratory Data Analysis" in prompt
    assert "Summarize missing data patterns" in prompt
    assert "sensor drift" in prompt
    assert "coastal regions" in prompt
    assert "Simulation assumes limited adaptation efforts." in prompt


@pytest.mark.parametrize("missing_field", ["dataset_name", "dataset_overview", "simulation_objective"])
def test_missing_required_metadata_raises_error(
    sample_metadata: dict[str, object], missing_field: str
) -> None:
    builder = PromptBuilder()
    incomplete = dict(sample_metadata)
    incomplete.pop(missing_field)

    with pytest.raises(ValueError):
        builder.build_prompt(
            analysis_type="hypothesis_generation",
            simulation_metadata=incomplete,
        )


def test_assumptions_must_be_iterable(sample_metadata: dict[str, object]) -> None:
    builder = PromptBuilder()
    sample_metadata["assumptions"] = "invalid"

    with pytest.raises(TypeError):
        builder.build_prompt(
            analysis_type="result_interpretation",
            simulation_metadata=sample_metadata,
        )
