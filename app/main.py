"""Public FastAPI contract and one-page PrivateText demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import ModelNotReadyError, PiiRedactor
from app.schemas import RedactionRequest, RedactionResponse

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="PrivateText API", version="0.1.0", description="TensorRT-optimized PII redaction.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.state.model = PiiRedactor()


def get_model() -> PiiRedactor:
    return app.state.model


def not_ready(error: ModelNotReadyError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(model: PiiRedactor = Depends(get_model)) -> dict[str, object]:
    return model.health()


@app.get("/metrics")
def metrics(model: PiiRedactor = Depends(get_model)) -> dict[str, object]:
    try:
        return model.metrics()
    except ModelNotReadyError as error:
        raise not_ready(error) from error


@app.post("/redact", response_model=RedactionResponse)
def redact(request: RedactionRequest, model: PiiRedactor = Depends(get_model)) -> dict[str, object]:
    try:
        return model.redact(request.text)
    except ModelNotReadyError as error:
        raise not_ready(error) from error
