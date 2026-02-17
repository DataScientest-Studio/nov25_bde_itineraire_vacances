from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List

class CtxResponse(BaseModel):
    relation: str
    rows: List[Dict[str, Any]]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relation": "silver.datatourisme_prime",
                "rows": [{"id": "DT_1", "name": "Musée", "lat": 48.85, "lon": 2.35}]
            }
        }
    )
