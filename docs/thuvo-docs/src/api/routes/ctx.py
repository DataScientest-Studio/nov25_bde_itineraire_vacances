import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, text

router = APIRouter(prefix="/ctx", tags=["context"])

# -------------------------------------------------------------------
# DB
# -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing (ex: postgresql+psycopg2://user:pass@db:5432/prime)")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# -------------------------------------------------------------------
# Sources -> relations candidates (ordre = priorité)
# -------------------------------------------------------------------
RELATIONS: Dict[str, List[str]] = {
    "dt": [
        "gold.mv_datatourisme_prime_serving",
        "gold.mv_datatourisme_prime",
        "gold.datatourisme_prime_serving",
        "gold.datatourisme_prime",
        "gold.mv_datatourisme_discover",
        "gold.datatourisme_discover",
        "silver.datatourisme_prime",
        "raw.datatourisme_prime",
    ],
    "ta": [
        "gold.mv_tripadvisor_discover",
        "silver.tripadvisor_discover",
        "raw.tripadvisor",
    ],
    "ab": [
        "gold.mv_airbnb_discover",
        "silver.airbnb_discover",
        "raw.airbnb",
    ],
}

# -------------------------------------------------------------------
# Pydantic schema (pour Swagger propre)
# -------------------------------------------------------------------
class CtxResponse(BaseModel):
    relation: str
    rows: List[Dict[str, Any]]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relation": "silver.datatourisme_prime",
                "rows": [
                    {"id": "DT_1", "name": "Musée", "lat": 48.85, "lon": 2.35}
                ],
            }
        }
    )

# -------------------------------------------------------------------
# SQL helpers
# -------------------------------------------------------------------
EXISTS_SQL = text("""
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema AND table_name = :name
    UNION ALL
    SELECT 1
    FROM information_schema.views
    WHERE table_schema = :schema AND table_name = :name
    LIMIT 1
""")

# anti injection basique : schema.table uniquement
REL_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$", re.I)

def _exists(relname: str) -> bool:
    if not REL_RE.match(relname):
        return False
    schema, name = relname.split(".", 1)
    with engine.connect() as conn:
        return conn.execute(EXISTS_SQL, {"schema": schema, "name": name}).first() is not None

def _first_existing(candidates: List[str]) -> Optional[str]:
    for rel in candidates:
        if _exists(rel):
            return rel
    return None

def _fetch_rows(rel: str, limit: int) -> List[Dict[str, Any]]:
    if not REL_RE.match(rel):
        raise HTTPException(status_code=400, detail=f"Invalid relation name: {rel}")

    try:
        q = text(f"SELECT * FROM {rel} LIMIT :limit")
        with engine.connect() as conn:
            result = conn.execute(q, {"limit": limit})
            return [dict(r._mapping) for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Route
# -------------------------------------------------------------------
@router.get("/{source}", response_model=CtxResponse)
def ctx(source: str, limit: int = 5000):
    source = source.lower().strip()

    if source not in RELATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source: {source}. Use one of {list(RELATIONS.keys())}"
        )

    rel = _first_existing(RELATIONS[source])
    if not rel:
        raise HTTPException(
            status_code=404,
            detail=f"No table/view found for source={source}. Candidates: {RELATIONS[source]}",
        )

    rows = _fetch_rows(rel, limit)
    return {"relation": rel, "rows": rows}
