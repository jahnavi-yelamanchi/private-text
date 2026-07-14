import pytest

from app.model import ModelNotReadyError, PiiRedactor


def test_missing_production_pointer_returns_clear_readiness_error(tmp_path) -> None:
    model = PiiRedactor(tmp_path)

    with pytest.raises(ModelNotReadyError, match="No promoted TensorRT artifact"):
        model.metrics()

    assert model.health()["status"] == "not_ready"
