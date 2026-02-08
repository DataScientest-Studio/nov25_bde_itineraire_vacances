from fastapi import APIRouter, Depends
from app.dependencies import get_db
from app.services.poi_service import poi_service
from app.models.poi import POIFilter, POIResponse, POI

router = APIRouter(prefix="/poi", tags=["poi"])

@router.post("/query", response_model=POIResponse)
def query_poi(filters: POIFilter, db=Depends(get_db)):
    rows = poi_service.get_filtered_pois(db, filters)
    pois = [POI(**row) for row in rows]
    return POIResponse(pois=pois)