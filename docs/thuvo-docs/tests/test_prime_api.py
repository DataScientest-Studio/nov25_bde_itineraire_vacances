from fastapi.testclient import TestClient

from api.main import app

pytest.skip("Skipping tests in personal branch", allow_module_level=True)

client = TestClient(app)


def test_itinerary_endpoint_exists():
    response = client.post("/itinerary", json={"dummy": "value"})
    assert response.status_code in (200, 422)
