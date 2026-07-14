from app.redaction import EntitySpan, non_overlapping, redact_text


def test_redaction_uses_original_offsets_from_right_to_left() -> None:
    text = "Email Ada at ada@example.com."
    spans = [
        EntitySpan("PERSON", "Ada", 6, 9, 0.98),
        EntitySpan("EMAIL", "ada@example.com", 13, 28, 0.99),
    ]

    redacted, selected = redact_text(text, spans)

    assert redacted == "Email [PERSON] at [EMAIL]."
    assert selected == spans


def test_overlapping_spans_keep_highest_confidence_prediction() -> None:
    text = "Call Ada Lovelace"
    spans = [
        EntitySpan("PERSON", "Ada", 5, 8, 0.65),
        EntitySpan("PERSON", "Ada Lovelace", 5, 17, 0.92),
    ]

    assert non_overlapping(spans, text) == [spans[1]]


def test_invalid_text_span_is_not_redacted() -> None:
    text = "Ada"
    invalid = EntitySpan("PERSON", "Not Ada", 0, 3, 0.99)

    assert redact_text(text, [invalid]) == (text, [])
