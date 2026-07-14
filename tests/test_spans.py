from app.spans import decode_bio_spans


def test_bio_tokens_become_one_character_accurate_span() -> None:
    text = "Email Ada Lovelace."
    spans = decode_bio_spans(
        text,
        offsets=[(0, 5), (6, 9), (10, 18), (18, 19)],
        labels=["O", "B-PERSON", "I-PERSON", "O"],
        confidences=[0.99, 0.90, 0.80, 0.99],
    )

    assert len(spans) == 1
    assert spans[0].text == "Ada Lovelace"
    assert spans[0].start == 6
    assert spans[0].end == 18
    assert round(spans[0].confidence, 2) == 0.85


def test_invalid_bio_continuation_starts_a_new_span() -> None:
    text = "Ada Paris"
    spans = decode_bio_spans(
        text,
        offsets=[(0, 3), (4, 9)],
        labels=["B-PERSON", "I-LOCATION"],
        confidences=[0.9, 0.8],
    )

    assert [span.type for span in spans] == ["PERSON", "LOCATION"]


def test_adjacent_restarted_labels_become_one_user_facing_span() -> None:
    text = "Sarah Johnson"
    spans = decode_bio_spans(
        text,
        offsets=[(0, 5), (6, 13)],
        labels=["B-PERSON", "B-PERSON"],
        confidences=[0.9, 0.8],
    )

    assert len(spans) == 1
    assert spans[0].text == "Sarah Johnson"
    assert spans[0].type == "PERSON"
