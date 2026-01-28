from fastapi import APIRouter, Depends
from api.models import ItineraryRequest, ItineraryResponse
from services.itinerary_service import ItineraryService

router = APIRouter(prefix="/itinerary")

def get_service():
    return ItineraryService(
        pois_path="../data/processed/merged_20260108_174125.parquet",
        osrm_url="http://localhost:5000"
    )

@router.post("/", response_model=ItineraryResponse)
def compute_itinerary(req: ItineraryRequest, service: ItineraryService = Depends(get_service)):
    itinerary = service.compute(req)
    return {"itinerary": itinerary}