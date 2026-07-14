"""Public-dataset loading and char-span to BIO-label preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

SOURCE_DATASET = "ai4privacy/pii-masking-200k"
SPLIT_SEED = 42
MAX_SAMPLES = 20_000

LABEL_ALIASES = {
    "PERSON": "PERSON",
    "PERSON_NAME": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    # ai4privacy uses compact labels in the released JSONL files. Keep these
    # mappings explicit so the model learns the entity types the API promises.
    "FIRSTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "LASTNAME": "PERSON",
    "PREFIX": "PERSON",
    "USERNAME": "PERSON",
    "ACCOUNTNAME": "PERSON",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "MOBILE_PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONEIMEI": "PHONE",
    "ADDRESS": "ADDRESS",
    "STREET_ADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "ORGANIZATION": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "COMPANYNAME": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "CITY": "LOCATION",
    "COUNTY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "ZIP_CODE": "LOCATION",
    "POSTCODE": "LOCATION",
    "DATE": "DATE",
    "DATE_TIME": "DATE",
    "DOB": "DATE",
    "ACCOUNT_NUMBER": "ACCOUNT_ID",
    "ACCOUNT_ID": "ACCOUNT_ID",
    "ACCOUNTNUMBER": "ACCOUNT_ID",
    "IBAN": "ACCOUNT_ID",
    "CREDIT_CARD": "ACCOUNT_ID",
    "CREDITCARDNUMBER": "ACCOUNT_ID",
    "CREDITCARDCVV": "ACCOUNT_ID",
    "SSN": "ACCOUNT_ID",
    "PASSPORT": "ACCOUNT_ID",
    "DRIVERS_LICENSE": "ACCOUNT_ID",
    "VEHICLEVIN": "ACCOUNT_ID",
    "VEHICLEVRM": "ACCOUNT_ID",
    "IP": "ACCOUNT_ID",
    "IPV4": "ACCOUNT_ID",
    "IPV6": "ACCOUNT_ID",
    "MAC": "ACCOUNT_ID",
    "BIC": "ACCOUNT_ID",
    "PIN": "ACCOUNT_ID",
    "MASKEDNUMBER": "ACCOUNT_ID",
}


@dataclass(frozen=True, slots=True)
class SourceSpan:
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SourceRecord:
    text: str
    spans: tuple[SourceSpan, ...]


def normalize_label(label: str) -> str | None:
    """Map source labels into the documented public API vocabulary."""

    normalized = label.strip().upper().replace(" ", "_").replace("-", "_")
    return LABEL_ALIASES.get(normalized)


def _as_mask(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("spans", value.get("entities", value.get("privacy_mask", [])))
    if not isinstance(value, list):
        raise ValueError("privacy_mask must contain a list of span records.")
    return [item for item in value if isinstance(item, dict)]


def record_from_row(row: dict[str, Any]) -> SourceRecord:
    """Read the documented ai4privacy text/mask fields defensively."""

    text = row.get("source_text") or row.get("text") or row.get("unmasked_text")
    if not isinstance(text, str) or not text:
        raise ValueError("Dataset row has no usable source text.")
    raw_spans = _as_mask(row.get("privacy_mask", row.get("entities", [])))
    spans: list[SourceSpan] = []
    for raw in raw_spans:
        raw_label = raw.get("label") or raw.get("type") or raw.get("entity")
        label = normalize_label(str(raw_label)) if raw_label is not None else None
        start, end = raw.get("start"), raw.get("end")
        if label and isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            spans.append(SourceSpan(label=label, start=start, end=end))
    if not spans:
        raise ValueError("Dataset row has no supported labeled spans.")
    return SourceRecord(text=text, spans=tuple(sorted(spans, key=lambda span: (span.start, span.end))))


def load_source_records(max_samples: int = MAX_SAMPLES) -> list[SourceRecord]:
    """Download a fixed shuffled public-only sample with no source data committed."""

    from datasets import load_dataset

    dataset = load_dataset(SOURCE_DATASET, split="train")
    dataset = dataset.shuffle(seed=SPLIT_SEED).select(range(min(max_samples, len(dataset))))
    records: list[SourceRecord] = []
    for row in dataset:
        try:
            records.append(record_from_row(dict(row)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not records:
        raise RuntimeError("No supported PII spans were found in the selected public dataset sample.")
    return records


def label_vocabulary(records: Iterable[SourceRecord]) -> list[str]:
    """Return a stable BIO vocabulary with outside label first."""

    entity_types = sorted({span.label for record in records for span in record.spans})
    return ["O", *(f"{prefix}-{entity_type}" for entity_type in entity_types for prefix in ("B", "I"))]


def split_records(records: list[SourceRecord]) -> dict[str, list[SourceRecord]]:
    """Create deterministic 80/10/10 splits without writing raw text to disk."""

    from sklearn.model_selection import train_test_split

    indices = list(range(len(records)))
    train_indices, held_out = train_test_split(indices, test_size=0.2, random_state=SPLIT_SEED)
    validation_indices, test_indices = train_test_split(held_out, test_size=0.5, random_state=SPLIT_SEED)
    return {
        "train": [records[index] for index in train_indices],
        "validation": [records[index] for index in validation_indices],
        "test": [records[index] for index in test_indices],
    }


def bio_labels_for_offsets(record: SourceRecord, offsets: Iterable[tuple[int, int]]) -> list[str]:
    """Assign BIO labels to tokenizer offsets; special/padding offsets become O."""

    labels: list[str] = []
    prior_span: SourceSpan | None = None
    for start, end in offsets:
        matching = next((span for span in record.spans if start < span.end and end > span.start and end > start), None)
        if matching is None:
            labels.append("O")
        elif matching == prior_span:
            labels.append(f"I-{matching.label}")
        else:
            labels.append(f"B-{matching.label}")
        prior_span = matching
    return labels
