"""Public request and response types for the PrivateText API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RedactionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, description="Unstructured text to scan for PII.")


class EntityResponse(BaseModel):
    type: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)


class RedactionResponse(BaseModel):
    redacted: str
    entities: list[EntityResponse]
    model_version: str
