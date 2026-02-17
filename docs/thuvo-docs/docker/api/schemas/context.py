from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional


# --------------------------------------------------
# Modèle métier strict
# --------------------------------------------------
class POI(BaseModel):
    source_id: Optional[str] = None
    name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    main_category: Optional[str] = None
    price_level: Optional[int] = None
    rating: Optional[float] = None


# --------------------------------------------------
# Réponse générique API (celle utilisée par /ctx/*)
# --------------------------------------------------
class ContextResponse(BaseModel):
    items: List[Dict[str, Any]]
    count: int

    # Exemple Swagger (sinon il affiche "string")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "source_id": "DT_0001",
                        "name": "Musée du Louvre",
                        "lat": 48.8606,
                        "lon": 2.3376,
                        "main_category": "Culture"
                    }
                ],
                "count": 1
            }
        }
    )
