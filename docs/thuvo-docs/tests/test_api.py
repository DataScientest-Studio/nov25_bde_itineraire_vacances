from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_itinerary_endpoint_exists():
    payload = {
        # payload minimal valide
        # adapte selon ce que /itinerary attend vraiment}

    response = client.post("/itinerary", json=payload)

    # Selon choix métier :
    # - 200 si payload OK
    # - 422 si payload incomplet (FastAPI validation)
    assert response.status_code in (200, 422)
