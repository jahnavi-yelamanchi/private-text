from fastapi.testclient import TestClient

from app.main import app, get_model


class FakeRedactor:
    def health(self):
        return {"status": "ready", "runtime": "fake"}

    def metrics(self):
        return {"test": {"entity_f1": 0.8}}

    def redact(self, text: str):
        return {
            "redacted": text.replace("Ada", "[PERSON]"),
            "entities": [{"type": "PERSON", "text": "Ada", "start": 0, "end": 3, "confidence": 0.99}],
            "model_version": "test-run",
        }


def test_redact_endpoint_returns_typed_response() -> None:
    app.dependency_overrides[get_model] = lambda: FakeRedactor()
    try:
        response = TestClient(app).post("/redact", json={"text": "Ada"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["redacted"] == "[PERSON]"
    assert response.json()["entities"][0]["start"] == 0


def test_redact_endpoint_rejects_blank_input() -> None:
    response = TestClient(app).post("/redact", json={"text": ""})

    assert response.status_code == 422


def test_demo_page_serves_the_redaction_workbench() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Redact private data" in response.text
    assert "privacy-illustration.png" in response.text
