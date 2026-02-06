from pydantic import BaseModel
from typing import List, Optional


class POI(BaseModel):
    poi_id: int
    nom_du_poi: str
    latitude: float
    longitude: float
    main_category: str
    sub_category: str
    h3_r7: str
    diversity_commune_norm: float
    final_score: float

class ItineraryRequest(BaseModel):
    pois: List[POI]
    days: int
    transport_mode: str = "walk"
    solver: str = "auto"
    latitude: float
    longitude: float


class ItineraryPOI(BaseModel):
    osrm_index: int
    poi_id: int
    cluster_id: int
    latitude: float
    longitude: float
    main_category: str
    sub_category: Optional[str]
    final_score: float
    order: int
    solver_used: str
    distance_from_prev: float
    duration_from_prev: float
    cumulative_distance: float
    cumulative_duration: float
    day_total_distance: float
    day_total_duration: float


class DayItinerary(BaseModel):
    day: int
    pois: List[ItineraryPOI]
    total_distance_km: float
    total_duration_min: float


class ItineraryResponse(BaseModel):
    itinerary: List[DayItinerary]
    trip_total_distance_km: float
    trip_total_duration_min: float
    optimizer: str
