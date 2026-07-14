# PrivateText

PrivateText is a deliberately small PII-redaction service. It fine-tunes a token-classification model, compiles the promoted model with TensorRT, and serves redacted text plus character-accurate entity spans through a FastAPI API.

## Status

Bootstrap complete. The repository is intentionally paused until its GitHub remote is connected.

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
