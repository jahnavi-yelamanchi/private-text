"""Modal GPU fine-tuning entry point for the PrivateText NER model."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from modal_app.data import (
    LABEL_ALIASES,
    MAX_SAMPLES,
    SOURCE_DATASET,
    SPLIT_SEED,
    SourceRecord,
    SourceSpan,
    bio_labels_for_offsets,
    label_vocabulary,
    load_source_records,
    split_records,
)
from modal_app.runtime import ARTIFACTS_PATH, app, cuda_image, model_volume

BASE_MODEL = "distilbert-base-uncased"


def _dataset_from_records(records: list[SourceRecord], tokenizer, label_to_id: dict[str, int]):
    """Tokenize source text and create a labels column from char-level PII spans."""

    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "text": [record.text for record in records],
            "spans": [[{"label": span.label, "start": span.start, "end": span.end} for span in record.spans] for record in records],
        }
    )

    def tokenize(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=256, return_offsets_mapping=True)
        label_rows = []
        for text, source_spans, offsets in zip(batch["text"], batch["spans"], encoded["offset_mapping"], strict=True):
            record = SourceRecord(text, tuple(SourceSpan(**span) for span in source_spans))
            labels = bio_labels_for_offsets(record, [tuple(offset) for offset in offsets])
            label_rows.append([label_to_id[label] if offset != (0, 0) else -100 for label, offset in zip(labels, offsets, strict=True)])
        encoded.pop("offset_mapping")
        encoded["labels"] = label_rows
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def _seqeval_metrics(label_names: list[str]):
    """Create Trainer-compatible entity-level metric calculation."""

    import numpy as np
    from seqeval.metrics import f1_score, precision_score, recall_score

    def compute(prediction):
        predictions = np.argmax(prediction.predictions, axis=-1)
        references = prediction.label_ids
        true_predictions = [
            [label_names[predicted] for predicted, reference in zip(row_predictions, row_references, strict=True) if reference != -100]
            for row_predictions, row_references in zip(predictions, references, strict=True)
        ]
        true_references = [
            [label_names[reference] for reference in row_references if reference != -100] for row_references in references
        ]
        precision = float(precision_score(true_references, true_predictions, zero_division=0))
        recall = float(recall_score(true_references, true_predictions, zero_division=0))
        return {
            "entity_f1": float(f1_score(true_references, true_predictions, zero_division=0)),
            "entity_precision": precision,
            "entity_recall": recall,
            "missed_pii_rate": 1.0 - recall,
            "false_redaction_rate": 1.0 - precision,
        }

    return compute


@app.function(image=cuda_image, gpu="T4", timeout=60 * 60, volumes={ARTIFACTS_PATH: model_volume})
def train() -> dict[str, object]:
    """Fine-tune one reproducible FP32 NER checkpoint and record held-out metrics."""

    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

    records = load_source_records(MAX_SAMPLES)
    splits = split_records(records)
    labels = label_vocabulary(records)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = _dataset_from_records(splits["train"], tokenizer, label_to_id)
    validation_dataset = _dataset_from_records(splits["validation"], tokenizer, label_to_id)
    test_dataset = _dataset_from_records(splits["test"], tokenizer, label_to_id)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/private-text-training",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=25,
            report_to="none",
            seed=SPLIT_SEED,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=_seqeval_metrics(labels),
    )
    trainer.train()
    test_metrics = {key.removeprefix("test_"): value for key, value in trainer.evaluate(test_dataset, metric_key_prefix="test").items()}

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_path = Path(ARTIFACTS_PATH) / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(run_path / "fp32"))
    tokenizer.save_pretrained(str(run_path / "fp32"))
    metrics = {
        "run_id": run_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL,
        "runtime": "pytorch-fp32",
        "source_dataset": SOURCE_DATASET,
        "max_samples": MAX_SAMPLES,
        "split_seed": SPLIT_SEED,
        "split_counts": {name: len(partition) for name, partition in splits.items()},
        "label_mapping": LABEL_ALIASES,
        "labels": labels,
        "test": test_metrics,
    }
    (run_path / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (run_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_volume.commit()
    return metrics


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(train.remote(), indent=2))
