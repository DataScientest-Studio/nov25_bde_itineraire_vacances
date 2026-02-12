import asyncio
import time

from fastapi import APIRouter, Depends
from app.models.itinerary import ItineraryRequest, ItineraryResponse
from app.dependencies import get_itinerary_service
from app.services.itinerary_service import ItineraryService
from app.monitoring import track_itinerary_request


router = APIRouter(prefix="/itinerary", tags=["itinerary"])


@router.post("/compute", response_model=ItineraryResponse)
async def compute_itinerary(
    req: ItineraryRequest,
    service: ItineraryService = Depends(get_itinerary_service)
):
    start_time = time.time()
    
    result = await service.compute_itinerary(
        pois=req.pois,
        days=req.days,
        transport_mode=req.transport_mode,
        solver=req.solver,
        start_lat=req.latitude,
        start_lon=req.longitude,
    )
    
    # Enregistrer les métriques de calcul d'itinéraire
    duration = time.time() - start_time
    pois_count = len(req.pois)
    
    track_itinerary_request(
        transport_mode=req.transport_mode,
        solver=req.solver,
        days=req.days,
        pois_count=pois_count,
        duration=duration
    )

    return {
        "itinerary": result["itinerary"],
        "trip_total_distance_km": result["trip_total_distance_km"],
        "trip_total_duration_min": result["trip_total_duration_min"],
        "optimizer": result["optimizer"]
    }

