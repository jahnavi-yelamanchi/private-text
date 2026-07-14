# PrivateText — TensorRT PII Redaction Service

## Goal

Build a small, public Modal-hosted service that redacts personally identifiable information (PII) from text. It will prove a complete NLP lifecycle: public-data fine-tuning, TensorRT FP16 optimization, GPU deployment, typed API design, and a focused web demo.

## Product contract

- `POST /redact` accepts `{"text":"..."}` and returns the redacted text, detected entities (`type`, `text`, `start`, `end`, `confidence`), and model version.
- `GET /health` reports model readiness; `GET /metrics` returns only measured evaluation and GPU-inference results.
- Detected spans are deduplicated and replaced from right to left with `[TYPE]`, preserving all other input text.

## Model and deployment

- Fine-tune `distilbert-base-uncased` for token classification on a reproducible public-only subset of `ai4privacy/pii-masking-200k`; document the source labels and exact mapping.
- Align BIO labels to subword tokens, evaluate entity-level F1 plus per-label metrics, and promote only a held-out evaluated run.
- Compile the promoted model directly with Torch-TensorRT FP16 for batch size 1 and dynamic sequence lengths 8–256; serve it from a CUDA Modal image, with tokenizer and post-processing in Python.
- Benchmark baseline PyTorch GPU and TensorRT GPU latency/throughput, then retain the promoted TensorRT artifact in a Modal Volume.

## Interface and quality bar

- Build one original Clay-inspired demo: cream canvas, near-black rounded typography, peach input editor, lavender output, teal metrics band, and a small privacy-themed illustration. It is a usable redaction surface, not a dashboard.
- Test label alignment, span reconstruction, overlap handling, replacement order, input validation, TensorRT artifact loading, and end-to-end API behavior.
- Keep raw dataset and generated artifacts out of Git. Publish reproducible commands, model provenance, measured results, and Docker/Modal deployment instructions in the README.

## Commit cadence

This bootstrap is committed before a GitHub remote is connected. After the user confirms `origin`, push this commit and make small, meaningful commits for the dataset pipeline, training, evaluation, TensorRT compilation, API, UI, tests, and deployment documentation.
