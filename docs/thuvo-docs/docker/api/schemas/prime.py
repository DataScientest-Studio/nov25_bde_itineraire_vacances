from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class PrimeRequest(BaseModel):
    # Adresse ancre
    anchor_numero: Optional[str] = None
    anchor_rue: str
    anchor_cp: str
    anchor_ville: str

    # Paramètres
    duree_sejour: int = Field(ge=1, le=30)
    rythme: Literal["Zen", "Normal", "Soutenu"] = "Normal"
    budget: Literal["Éco", "Moyen", "Confort", "Premium"] = "Moyen"
    categories: List[str] = []

    distance_max_km: float = Field(ge=0.5, le=30.0)

class PrimeItem(BaseModel):
    jour: int
    slot: str
    type: str
    name: str
    category: Optional[str] = None
    category_type: Optional[str] = None
    cuisine_continent: Optional[str] = None
    price_level: Optional[str] = None

    score: Optional[float] = None
    distance_to_anchor_km: Optional[float] = None
    distance_to_poi_central_km: Optional[float] = None

    lat: Optional[float] = None
    lon: Optional[float] = None
    address: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None

class PrimeResponse(BaseModel):
    anchor_lat: Optional[float] = None
    anchor_lon: Optional[float] = None
    itinerary: List[PrimeItem] = []
    warnings: List[str] = []
