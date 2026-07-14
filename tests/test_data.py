from modal_app.data import SourceRecord, SourceSpan, bio_labels_for_offsets, normalize_label, record_from_row


def test_source_labels_are_normalized_into_api_types() -> None:
    assert normalize_label("email_address") == "EMAIL"
    assert normalize_label("credit-card") == "ACCOUNT_ID"
    assert normalize_label("unknown") is None


def test_dataset_row_reads_char_offsets() -> None:
    row = {
        "source_text": "Email Ada at ada@example.com",
        "privacy_mask": [{"label": "PERSON_NAME", "start": 6, "end": 9}],
    }

    record = record_from_row(row)

    assert record == SourceRecord("Email Ada at ada@example.com", (SourceSpan("PERSON", 6, 9),))


def test_char_spans_become_bio_labels() -> None:
    record = SourceRecord("Ada Lovelace", (SourceSpan("PERSON", 0, 12),))

    assert bio_labels_for_offsets(record, [(0, 3), (4, 12), (0, 0)]) == ["B-PERSON", "I-PERSON", "O"]
