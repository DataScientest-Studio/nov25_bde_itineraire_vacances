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

class ItineraryDay(BaseModel):
    day: int
    pois: List[Dict]
    total_distance_km: float
    total_duration_min: float

class ItineraryResponse(BaseModel):
    itinerary: List[ItineraryDay]