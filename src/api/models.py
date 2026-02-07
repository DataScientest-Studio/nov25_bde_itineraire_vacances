from pydantic import BaseModel, Field
from typing import List

class CategoriesRequest(BaseModel):
    categories_list: List[str]

class StartPoint(BaseModel):
    lat: float = Field(..., ge=-180, le=180)
    lon: float = Field(..., ge=-90, le=90)
    radius: int = Field(3000, ge=1000, le=30000)


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
    adresse: str
    code_postal: str
    commune: str
    departement: str
    region: str
    contacts_du_poi: str
    final_score: float
    order: int
    solver_used: str
    distance_from_prev_km: float
    duration_from_prev_min: float
    cumulative_distance_km: float
    cumulative_duration_min: float
    day_total_distance_km: float
    day_total_duration_min: float


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
