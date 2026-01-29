from pydantic import BaseModel
from typing import List, Dict

class StartPoint(BaseModel):
    lat: float
    lon: float

class ItineraryRequest(BaseModel):
    commune: str
    main_categories: List[str]
    sub_categories: List[str] = []
    min_score: float
    nb_days: int
    start: StartPoint
    transport_mode: str = "walk"
    solver: str = "auto"

class POI(BaseModel):
    osrm_index: int
    poi_id: int
    cluster_id: int
    latitude: float
    longitude: float
    main_category: str
    sub_category: str
    final_score: float
    order: int
    solver_used: str
    distance_from_prev: float
    duration_from_prev: float
    cumulative_distance: float
    cumulative_duration: float
    day_total_distance: float
    day_total_duration: float


class ItineraryDay(BaseModel):
    day: int
    pois: List[POI]
    total_distance_km: float
    total_duration_min: float

class ItineraryResponse(BaseModel):
    itinerary: List[ItineraryDay]
    trip_total_distance_km: float
    trip_total_duration_min: float
    optimizer: str