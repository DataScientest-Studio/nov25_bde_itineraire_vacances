from pydantic import BaseModel
from typing import Optional, List

class POIFilter(BaseModel):
    commune: str
    latitude: float
    longitude: float
    main_category: List[str]
    sub_category: Optional[List[str]]
    radius: int
    days: int

class POI(BaseModel):
    poi_id: int
    nom_du_poi: str
    latitude: float
    longitude: float
    main_category: str
    sub_category: Optional[str]
    h3_r7: str
    diversity_commune_norm: float
    final_score: float
    #commune: str

class POIResponse(BaseModel):
    pois: List[POI]