"""Span validation and deterministic PII redaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """One model prediction expressed against the original input string."""

    type: str
    text: str
    start: int
    end: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("Entity type cannot be empty.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Entity span offsets must form a non-empty range.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Entity confidence must be between zero and one.")


def non_overlapping(spans: Iterable[EntitySpan], text: str) -> list[EntitySpan]:
    """Keep valid, highest-confidence spans without allowing overlaps.

    Selection is deterministic: confidence wins first, then longer spans, then the
    earliest source offset. The result is returned in source order for API clients.
    """

    candidates = [span for span in spans if span.end <= len(text) and text[span.start : span.end] == span.text]
    selected: list[EntitySpan] = []
    for candidate in sorted(candidates, key=lambda span: (-span.confidence, -(span.end - span.start), span.start)):
        if all(candidate.end <= accepted.start or candidate.start >= accepted.end for accepted in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda span: (span.start, span.end))


def redact_text(text: str, spans: Iterable[EntitySpan]) -> tuple[str, list[EntitySpan]]:
    """Replace selected spans right-to-left so original offsets stay valid."""

    selected = non_overlapping(spans, text)
    redacted = text
    for span in reversed(selected):
        redacted = f"{redacted[:span.start]}[{span.type}]{redacted[span.end:]}"
    return redacted, selected
