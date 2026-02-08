from pydantic import BaseModel
from typing import List, Optional


class POI(BaseModel):
    poi_id: int
    nom_du_poi: Optional[str] = None
    description: Optional[str] = None
    adresse: Optional[str] = None
    latitude: float
    longitude: float
    main_category: str
    sub_category: str
    contact_phone: Optional[str] = None
    contact_mail: Optional[str] = None
    contact_website: Optional[str] = None
    itineraire: Optional[bool] = None
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
    cluster_id: int
    poi_id: int
    nom_du_poi: Optional[str] = None
    description: Optional[str] = None
    adresse: Optional[str] = None
    latitude: float
    longitude: float
    main_category: str
    sub_category: Optional[str]
    contact_phone: Optional[str] = None
    contact_mail: Optional[str] = None
    contact_website: Optional[str] = None
    itineraire: Optional[bool] = None
    final_score: float
    order: int
    solver_used: str
    distance_from_prev_km: float
    duration_from_prev_min: float
    cumulative_distance_km: float
    cumulative_duration_min: float
    day_total_distance_km: float
    day_total_duration_min: float


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
