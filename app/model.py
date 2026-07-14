"""Lazy TensorRT-backed token-classification runtime for PrivateText."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.redaction import EntitySpan, redact_text
from app.spans import decode_bio_spans


class ModelNotReadyError(RuntimeError):
    """Raised when a promoted TensorRT artifact is unavailable to the API."""


class PiiRedactor:
    """Load the promoted TensorRT artifact only when an inference call needs it."""

    def __init__(self, artifacts_path: str | Path | None = None) -> None:
        self.artifacts_path = Path(artifacts_path or os.environ.get("PRIVATE_TEXT_ARTIFACTS_PATH", "/models"))
        self._module: Any | None = None
        self._tokenizer: Any | None = None
        self._labels: list[str] = []
        self._model_version: str | None = None
        self._metrics: dict[str, object] | None = None

    def _ensure_loaded(self) -> None:
        if self._module is not None:
            return
        pointer_path = self.artifacts_path / "production.json"
        if not pointer_path.exists():
            raise ModelNotReadyError("No promoted TensorRT artifact is available yet.")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = pointer.get("run_id")
        artifact_name = pointer.get("artifact")
        if not isinstance(run_id, str) or not isinstance(artifact_name, str):
            raise ModelNotReadyError("The production artifact pointer is invalid.")
        run_path = self.artifacts_path / "runs" / run_id
        artifact_path = run_path / artifact_name
        labels_path = run_path / "labels.json"
        metrics_path = run_path / "metrics.json"
        if not artifact_path.exists() or not labels_path.exists() or not metrics_path.exists():
            raise ModelNotReadyError("The promoted TensorRT artifact is incomplete.")

        try:
            import torch
            import torch_tensorrt  # noqa: F401  # Registers TensorRT TorchScript operators.
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on deployment image
            raise ModelNotReadyError("TensorRT serving dependencies are not installed.") from error

        if not torch.cuda.is_available():  # pragma: no cover - depends on deployment image
            raise ModelNotReadyError("PrivateText requires a CUDA GPU to serve its TensorRT artifact.")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(labels, dict):
            labels = [labels[str(index)] for index in range(len(labels))]
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ModelNotReadyError("The promoted label vocabulary is invalid.")

        self._module = torch.jit.load(str(artifact_path), map_location="cuda").eval()
        self._tokenizer = AutoTokenizer.from_pretrained(run_path / "fp32")
        self._labels = labels
        self._model_version = run_id
        self._metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def health(self) -> dict[str, object]:
        try:
            self._ensure_loaded()
        except ModelNotReadyError as error:
            return {"status": "not_ready", "detail": str(error)}
        return {"status": "ready", "model_version": self._model_version, "runtime": "torch-tensorrt-fp16"}

    def metrics(self) -> dict[str, object]:
        self._ensure_loaded()
        assert self._metrics is not None
        return self._metrics

    def _infer(self, text: str) -> list[EntitySpan]:
        self._ensure_loaded()
        assert self._module is not None and self._tokenizer is not None
        import torch

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
        input_ids = encoded["input_ids"].to(device="cuda", dtype=torch.int32)
        attention_mask = encoded["attention_mask"].to(device="cuda", dtype=torch.int32)
        with torch.inference_mode():
            output = self._module(input_ids, attention_mask)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities = torch.softmax(logits[0].float(), dim=-1).cpu()
        label_ids = probabilities.argmax(dim=-1).tolist()
        confidences = probabilities.max(dim=-1).values.tolist()
        labels = [self._labels[index] for index in label_ids]
        return decode_bio_spans(text, offsets, labels, confidences)

    def redact(self, text: str) -> dict[str, object]:
        spans = self._infer(text)
        redacted, selected = redact_text(text, spans)
        return {
            "redacted": redacted,
            "entities": [
                {
                    "type": span.type,
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "confidence": round(span.confidence, 4),
                }
                for span in selected
            ],
            "model_version": self._model_version,
        }
