from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_itinerary_ok():
    payload = {
        # payload minimal valide
    }

    response = client.post("/itinerary", json=payload)
    assert response.status_code == 200
