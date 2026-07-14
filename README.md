# PrivateText

PrivateText is a deliberately small PII-redaction service. It fine-tunes a token-classification model, compiles the promoted model directly with Torch-TensorRT FP16, and serves redacted text plus character-accurate entity spans through a FastAPI API.

## Status

Live service: [PrivateText on Modal](https://jahnavi-yelamanchi--private-text-pii-redaction-fastapi-app.modal.run)

## Target API

```json
POST /redact
{"text":"Email Jahnavi at jahnavi@example.com."}
```

```json
{
  "redacted": "Email [PERSON] at [EMAIL].",
  "entities": [
    {"type": "PERSON", "text": "Jahnavi", "start": 6, "end": 13, "confidence": 0.98},
    {"type": "EMAIL", "text": "jahnavi@example.com", "start": 17, "end": 36, "confidence": 0.99}
  ],
  "model_version": "20260714T000000Z"
}
```

See [PLAN.md](PLAN.md) for the approved implementation plan.

## Architecture

```text
public PII text
      |
      v
FastAPI /redact ──> tokenizer + TensorRT FP16 token classifier ──> BIO spans
      |                                                               |
      +-------------------- redacted text + char offsets <------------+

Modal GPU fine-tuning ──> evaluated FP32 checkpoint ──> Torch-TensorRT FP16 artifact
                                                                    |
                                                                    v
                                                             Modal Volume promotion
```

## Train, optimize, and deploy

```bash
modal run modal_app/train.py
modal run modal_app/export.py --run-id <training-run-id>
modal deploy modal_app/service.py
```

See [deployment instructions](docs/deployment.md) for the artifact lifecycle and local CUDA container command.

The TensorRT artifact uses a fixed padded 256-token profile. This avoids the dynamic-profile incompatibility between the current Torch-TensorRT release and DistilBERT's learned positional embeddings; API input is still capped at 10,000 characters and truncated safely by the tokenizer.

## Dataset and label policy

Training reads a reproducible, seed-42 sample of up to 20,000 records from [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k). Raw data is downloaded only inside the training environment and is never committed.

The project preserves the training run's exact source-label mapping in `metrics.json`, while normalizing supported labels into this stable public API vocabulary:

| API type | Source examples |
| --- | --- |
| `PERSON` | `PERSON`, `PERSON_NAME`, `FIRST_NAME`, `LAST_NAME` |
| `EMAIL` | `EMAIL`, `EMAIL_ADDRESS` |
| `PHONE` | `PHONE`, `PHONE_NUMBER`, `MOBILE_PHONE` |
| `ADDRESS` | `ADDRESS`, `STREET_ADDRESS` |
| `ORGANIZATION` | `ORGANIZATION`, `ORGANISATION`, `COMPANY` |
| `LOCATION` | `LOCATION`, `CITY`, `STATE`, `COUNTRY`, `ZIP_CODE` |
| `DATE` | `DATE`, `DATE_TIME` |
| `ACCOUNT_ID` | `ACCOUNT_NUMBER`, `IBAN`, `CREDIT_CARD`, `SSN`, `PASSPORT` |

## Measured results only

The `/metrics` endpoint is unavailable until a held-out test run is evaluated and promoted. It then reports entity F1, precision/recall-derived missed-PII and false-redaction rates, plus baseline PyTorch GPU and TensorRT FP16 GPU P50/P95 latency, throughput, and artifact size. The interface deliberately shows placeholders beforehand rather than invented performance claims.

## Measured run

The promoted run `20260714T185931Z` trained on 5,428 records, validated on 678, and evaluated on 679 held-out public-dataset records. The requested 20,000-record sample yielded 6,785 records with supported, offset-valid PII spans after filtering.

| Held-out metric | Result |
| --- | ---: |
| Entity F1 | 0.8144 |
| Entity precision | 0.7664 |
| Entity recall | 0.8688 |
| Missed-PII rate | 13.12% |
| False-redaction rate | 23.36% |

| GPU artifact | Size | P50 latency | P95 latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | — | 7.528 ms | 7.766 ms | 132.84/s |
| TensorRT FP16 | 177.9 MB | 1.553 ms | 1.616 ms | 643.79/s |

The promoted artifact can currently emit `ACCOUNT_ID`, `DATE`, `EMAIL`, and `LOCATION` because those were the supported labels present in this filtered training sample. The API schema remains stable for future runs with broader public-data coverage.

## Test

```bash
python -m pip install "fastapi>=0.115,<1.0" "httpx>=0.27,<1.0" "pytest>=8.0,<9.0"
python -m pytest -q
```
