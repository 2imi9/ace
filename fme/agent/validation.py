"""Validation and guardrail utilities for ACE agent responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableSequence, Sequence
from uuid import uuid4

_CONSTRAINT_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)\s*(?P<op>==|>=|<=|~=|!=)\s*(?P<value>[^#\s]+)")


@dataclass
class ValidationResult:
    """Outcome of a validation pass."""

    is_valid: bool
    issues: List[str]
    final_response: str

    @property
    def transformed_response(self) -> str | None:
        """Return the transformed response when a modification was required."""

        return None if self.is_valid else self.final_response


class VerificationLayer:
    """Apply guardrails and validation checks to LLM outputs."""

    def __init__(
        self,
        constraints: Iterable[str] | None = None,
        *,
        banned_phrases: Mapping[str, str] | None = None,
        citation_required: bool = True,
        log_dir: str | Path = "logs/experiments",
        refusal_template: str | None = None,
    ) -> None:
        self.constraints = [self._normalise_constraint(c) for c in (constraints or self._load_default_constraints())]
        self.banned_phrases = dict(
            banned_phrases
            or {
                "hack": "Request references hacking instructions.",
                "malware": "Malware content is disallowed.",
            }
        )
        self.citation_required = citation_required
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.refusal_template = (
            refusal_template
            or "I’m sorry, but I cannot provide that because: {reasons}."
        )
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self._log_path = self.log_dir / f"session-{timestamp}-{uuid4().hex}.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        prompt: str,
        response: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate ``response`` given ``prompt`` and optional context."""

        issues: List[str] = []
        context = context or {}

        issues.extend(self._check_banned(prompt, "prompt"))
        issues.extend(self._check_banned(response, "response"))
        issues.extend(self._check_constraints(response))
        issues.extend(self._check_expected_facts(response, context))

        if self.citation_required and not self._contains_citation(response):
            issues.append("Response is missing the required citation markup.")

        is_valid = not issues
        final_response = response if is_valid else self._format_refusal(issues)
        result = ValidationResult(is_valid=is_valid, issues=issues, final_response=final_response)

        self._log_interaction(prompt, response, final_response, context, result)
        return result

    @property
    def log_path(self) -> Path:
        """Return the path to the JSONL experiment log."""

        return self._log_path

    # ------------------------------------------------------------------
    # Guardrail helpers
    # ------------------------------------------------------------------
    def _check_banned(self, text: str, source: str) -> List[str]:
        lowered = text.lower()
        violations = []
        for phrase, reason in self.banned_phrases.items():
            if phrase.lower() in lowered:
                violations.append(f"{source.capitalize()} triggered banned phrase '{phrase}': {reason}")
        return violations

    def _check_constraints(self, response: str) -> List[str]:
        if not self.constraints:
            return []
        lowered = response.lower()
        violations: List[str] = []
        for constraint in self.constraints:
            match = _CONSTRAINT_PATTERN.match(constraint)
            if not match:
                continue
            package = match.group("name")
            operator = match.group("op")
            value = match.group("value")
            if package.lower() in lowered:
                spec = f"{operator}{value}"
                if spec not in response:
                    violations.append(
                        f"Response mentions '{package}' but does not respect constraint '{constraint}'."
                    )
        return violations

    def _check_expected_facts(
        self,
        response: str,
        context: Mapping[str, Any],
    ) -> List[str]:
        facts = self._extract_expected_facts(context)
        if not facts:
            return []
        violations: List[str] = []
        for fact in facts:
            if fact and fact not in response:
                violations.append(
                    f"Response is missing expected fact '{fact}' derived from ACE outputs or constraints."
                )
        return violations

    def _contains_citation(self, response: str) -> bool:
        return bool(re.search(r"\[(?:citation|cite):[^\]]+\]", response))

    def _format_refusal(self, issues: Sequence[str]) -> str:
        reasons = "; ".join(issues)
        return self.refusal_template.format(reasons=reasons)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def _extract_expected_facts(self, context: Mapping[str, Any]) -> List[str]:
        candidates: List[str] = []
        keys_of_interest = {"expected_facts", "known_facts"}
        for key in keys_of_interest:
            values = context.get(key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                candidates.extend(str(value) for value in values if self._is_fact_value(value))
        ace_outputs = context.get("ace_outputs")
        if isinstance(ace_outputs, Mapping):
            for key in ("claims", "facts", "summary"):
                values = ace_outputs.get(key)
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    candidates.extend(str(value) for value in values if self._is_fact_value(value))
                elif self._is_fact_value(values):
                    candidates.append(str(values))
        return candidates

    def _is_fact_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, int, float)):
            text = str(value)
            return bool(text.strip()) and len(text.split()) <= 25
        return False

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_interaction(
        self,
        prompt: str,
        response: str,
        final_response: str,
        context: Mapping[str, Any],
        result: ValidationResult,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "raw_response": response,
            "final_response": final_response,
            "issues": list(result.issues),
            "is_valid": result.is_valid,
            "context": self._make_json_safe(context),
        }
        with self._log_path.open("a", encoding="utf-8") as stream:
            json.dump(entry, stream, default=str)
            stream.write("\n")

    def _make_json_safe(self, context: Mapping[str, Any]) -> Any:
        try:
            json.dumps(context)
            return context
        except TypeError:
            return json.loads(json.dumps(self._stringify(context)))

    def _stringify(self, item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: self._stringify(value) for key, value in item.items()}
        if isinstance(item, MutableSequence):
            return [self._stringify(value) for value in item]
        if isinstance(item, (set, frozenset, tuple)):
            return [self._stringify(value) for value in item]
        return item if isinstance(item, (int, float, str, bool, type(None))) else str(item)

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------
    def _load_default_constraints(self) -> List[str]:
        constraints_file = Path("constraints.txt")
        if not constraints_file.exists():
            return []
        contents = []
        for line in constraints_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            contents.append(line)
        return contents

    def _normalise_constraint(self, constraint: str) -> str:
        return constraint.strip()

