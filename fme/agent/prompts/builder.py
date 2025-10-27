"""Utilities for constructing prompts from templates and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

from .templates import (
    BASE_BRIEFING_TEMPLATE,
    DEFAULT_ANALYSIS_TEMPLATE,
    SPECIALIZED_ANALYSIS_TEMPLATES,
)


@dataclass(slots=True)
class PromptBuilder:
    """Render prompt templates with dataset and simulation metadata."""

    base_template: str = BASE_BRIEFING_TEMPLATE
    analysis_templates: MutableMapping[str, str] = field(
        default_factory=lambda: dict(SPECIALIZED_ANALYSIS_TEMPLATES)
    )
    default_analysis_template: str = DEFAULT_ANALYSIS_TEMPLATE

    REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset(
        {
            "dataset_name",
            "dataset_overview",
            "simulation_objective",
            "assumptions",
        }
    )

    def build_prompt(
        self,
        analysis_type: str,
        simulation_metadata: Mapping[str, object],
        variable_descriptions: Sequence[Mapping[str, object]] | None = None,
        requested_outputs: Sequence[str] | None = None,
        uncertainty_info: Mapping[str, object] | None = None,
        user_request: str | None = None,
        additional_context: str | None = None,
    ) -> str:
        """Render the configured template with metadata and analysis directives."""

        self._validate_metadata(simulation_metadata)
        assumption_lines = self._format_assumptions(simulation_metadata.get("assumptions"))
        metadata_section = self._format_simulation_metadata(simulation_metadata)
        variable_section = self._format_variables(variable_descriptions or [])
        uncertainty_section = self._format_uncertainty(uncertainty_info)
        outputs_section = self._format_outputs(requested_outputs)
        analysis_focus = self._select_analysis_template(analysis_type)

        analysis_label = analysis_type.replace("_", " ").strip()
        context = {
            "analysis_type_display": analysis_label.title() or "Analysis",
            "dataset_name": simulation_metadata["dataset_name"],
            "dataset_overview": simulation_metadata["dataset_overview"],
            "simulation_objective": simulation_metadata["simulation_objective"],
            "simulation_metadata_section": metadata_section,
            "assumption_bullets": assumption_lines,
            "variable_section": variable_section,
            "uncertainty_section": uncertainty_section,
            "analysis_focus": analysis_focus,
            "requested_outputs": outputs_section,
            "user_request": user_request or "No specific user request provided.",
            "additional_context": additional_context or "No additional context provided.",
        }

        return self.base_template.format_map(context)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _validate_metadata(self, metadata: Mapping[str, object]) -> None:
        missing = [field for field in self.REQUIRED_METADATA_FIELDS if field not in metadata]
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(f"Missing required simulation metadata fields: {formatted}")

        assumptions = metadata.get("assumptions")
        if not isinstance(assumptions, Iterable) or isinstance(assumptions, (str, bytes)):
            raise TypeError("'assumptions' must be an iterable of assumption strings.")

    def _format_assumptions(self, assumptions: object | None) -> str:
        if not assumptions:
            return "- No explicit assumptions provided."
        return "\n".join(f"- {assumption}" for assumption in assumptions)

    def _format_simulation_metadata(self, metadata: Mapping[str, object]) -> str:
        exclusions = self.REQUIRED_METADATA_FIELDS | {"assumptions"}
        lines = []
        for key, value in metadata.items():
            if key in exclusions:
                continue
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        if not lines:
            return "- No additional simulation metadata provided."
        return "\n".join(lines)

    def _format_variables(
        self, variable_descriptions: Sequence[Mapping[str, object]]
    ) -> str:
        if not variable_descriptions:
            return "- No variable descriptions supplied."
        lines = []
        for variable in variable_descriptions:
            name = variable.get("name", "unknown")
            description = variable.get("description", "No description provided")
            dtype = variable.get("type")
            unit = variable.get("unit")
            role = variable.get("role")
            details = [f"**Description:** {description}"]
            if dtype:
                details.append(f"**Type:** {dtype}")
            if unit:
                details.append(f"**Unit:** {unit}")
            if role:
                details.append(f"**Role:** {role}")
            lines.append(f"- **{name}:** " + "; ".join(details))
        return "\n".join(lines)

    def _format_uncertainty(self, info: Mapping[str, object] | None) -> str:
        if not info:
            return "- No uncertainty information supplied."
        lines = []
        sources = info.get("sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            lines.append("- **Sources:** " + ", ".join(str(item) for item in sources))
        elif sources:
            lines.append(f"- **Sources:** {sources}")
        confidence = info.get("confidence_level")
        if confidence is not None:
            lines.append(f"- **Confidence Level:** {confidence}")
        notes = info.get("notes")
        if notes:
            lines.append(f"- **Notes:** {notes}")
        other_keys = {
            key: value
            for key, value in info.items()
            if key not in {"sources", "confidence_level", "notes"}
        }
        for key, value in other_keys.items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        if not lines:
            return "- No uncertainty information supplied."
        return "\n".join(lines)

    def _format_outputs(self, outputs: Sequence[str] | None) -> str:
        if not outputs:
            return "- Provide high-level findings and actionable recommendations."
        return "\n".join(f"- {item}" for item in outputs)

    def _select_analysis_template(self, analysis_type: str) -> str:
        key = analysis_type.strip().lower().replace(" ", "_")
        template = self.analysis_templates.get(key)
        if template is None:
            return self.default_analysis_template
        return template
