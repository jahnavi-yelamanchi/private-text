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
| `PERSON` | `PERSON`, `PERSON_NAME`, `FIRST_NAME`, `LAST_NAME`, `FIRSTNAME`, `MIDDLENAME`, `LASTNAME`, `PREFIX` |
| `EMAIL` | `EMAIL`, `EMAIL_ADDRESS` |
| `PHONE` | `PHONE`, `PHONE_NUMBER`, `MOBILE_PHONE`, `PHONENUMBER`, `PHONEIMEI` |
| `ADDRESS` | `ADDRESS`, `STREET_ADDRESS`, `STREET`, `BUILDINGNUMBER`, `SECONDARYADDRESS`, `ZIPCODE` |
| `ORGANIZATION` | `ORGANIZATION`, `ORGANISATION`, `COMPANY`, `COMPANYNAME` |
| `LOCATION` | `LOCATION`, `CITY`, `COUNTY`, `STATE`, `COUNTRY`, `ZIP_CODE` |
| `DATE` | `DATE`, `DATE_TIME`, `DOB` |
| `ACCOUNT_ID` | `ACCOUNT_NUMBER`, `ACCOUNTNUMBER`, `IBAN`, `CREDIT_CARD`, `CREDITCARDNUMBER`, `SSN`, `PASSPORT` |

## Measured results only

The `/metrics` endpoint is unavailable until a held-out test run is evaluated and promoted. It then reports entity F1, precision/recall-derived missed-PII and false-redaction rates, plus baseline PyTorch GPU and TensorRT FP16 GPU P50/P95 latency, throughput, and artifact size. The interface deliberately shows placeholders beforehand rather than invented performance claims.

## Measured run

The promoted run `20260714T191813Z` trained on 14,737 records, validated on 1,842, and evaluated on 1,843 held-out public-dataset records. The fixed 20,000-record sample yielded 18,422 records with supported, offset-valid PII spans after filtering.

| Held-out metric | Result |
| --- | ---: |
| Entity F1 | 0.9011 |
| Entity precision | 0.8811 |
| Entity recall | 0.9220 |
| Missed-PII rate | 7.80% |
| False-redaction rate | 11.89% |

| GPU artifact | Size | P50 latency | P95 latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | — | 7.210 ms | 7.739 ms | 138.70/s |
| TensorRT FP16 | 178.0 MB | 1.520 ms | 1.543 ms | 657.88/s |

The promoted artifact emits `ACCOUNT_ID`, `ADDRESS`, `DATE`, `EMAIL`, `LOCATION`, `ORGANIZATION`, `PERSON`, and `PHONE` from the evaluated run. The API schema remains stable for future runs with broader public-data coverage.

## Test

```bash
python -m pip install "fastapi>=0.115,<1.0" "httpx>=0.27,<1.0" "pytest>=8.0,<9.0"
python -m pytest -q
```
