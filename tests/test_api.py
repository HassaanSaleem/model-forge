from fastapi.testclient import TestClient

from model_forge.summarizer.api import create_app
from model_forge.summarizer.pool import ModelPool


class StubModel:
    def summarize(self, text: str) -> str:
        return f"summary of: {text[:20]}"

    def summarize_batch(self, texts: list[str]) -> list[str]:
        return [self.summarize(t) for t in texts]


def client(pool_size: int = 2) -> TestClient:
    return TestClient(create_app(ModelPool(pool_size, StubModel)))


def test_summarize_endpoint():
    response = client().post("/summarize", json={"text": "ERROR db connection lost"})
    assert response.status_code == 200
    assert response.json()["summary"].startswith("summary of:")


def test_batch_summarize_endpoint():
    response = client().post("/batch_summarize", json={"texts": ["a", "b", "c"]})
    assert response.status_code == 200
    assert len(response.json()["summaries"]) == 3


def test_health_check_ok():
    response = client().get("/health_check")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "pool_size": 2}


def test_health_check_reports_saturation():
    pool = ModelPool(1, StubModel)
    app = create_app(pool)
    with TestClient(app) as test_client:
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(pool.acquire())  # hold the only replica
        response = test_client.get("/health_check")
        assert response.status_code == 503
        assert response.json()["status"] == "saturated"
        pool.release()
        loop.close()


def test_validation_rejects_bad_payload():
    response = client().post("/summarize", json={"wrong_field": "x"})
    assert response.status_code == 422
