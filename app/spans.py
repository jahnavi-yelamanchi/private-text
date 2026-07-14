"""Convert token-classification predictions into character-level entity spans."""

from __future__ import annotations

from collections.abc import Sequence

from app.redaction import EntitySpan


def _parts(label: str) -> tuple[str, str | None]:
    if label == "O":
        return "O", None
    prefix, separator, entity_type = label.partition("-")
    if separator and prefix in {"B", "I"} and entity_type:
        return prefix, entity_type
    return "B", label


def decode_bio_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    labels: Sequence[str],
    confidences: Sequence[float],
) -> list[EntitySpan]:
    """Merge token predictions using tokenizer offset mappings and BIO labels."""

    if not (len(offsets) == len(labels) == len(confidences)):
        raise ValueError("Offsets, labels, and confidences must have equal lengths.")

    entities: list[EntitySpan] = []
    active_type: str | None = None
    active_start = 0
    active_end = 0
    active_scores: list[float] = []

    def flush() -> None:
        nonlocal active_type, active_scores
        if active_type is None:
            return
        entities.append(
            EntitySpan(
                type=active_type,
                text=text[active_start:active_end],
                start=active_start,
                end=active_end,
                confidence=sum(active_scores) / len(active_scores),
            )
        )
        active_type = None
        active_scores = []

    for (start, end), label, confidence in zip(offsets, labels, confidences, strict=True):
        # Special tokens and padding generally have a 0,0 offset.
        if end <= start or start < 0 or end > len(text):
            flush()
            continue
        prefix, entity_type = _parts(label)
        # The source dataset can split one real-world value across compact
        # labels (for example BUILDINGNUMBER + STREET), and the fine-tuned
        # model can restart BIO at a whitespace boundary. Preserve a single
        # user-facing span when same-type tokens are contiguous or separated
        # only by one space.
        continuation = entity_type == active_type and prefix in {"B", "I"} and start <= active_end + 1
        if prefix == "O" or entity_type is None:
            flush()
        elif continuation:
            active_end = end
            active_scores.append(confidence)
        else:
            flush()
            active_type = entity_type
            active_start = start
            active_end = end
            active_scores = [confidence]
    flush()
    return entities
