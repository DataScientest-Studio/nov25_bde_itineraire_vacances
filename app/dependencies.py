from app.core.database import db_manager
import os

from fastapi import Depends

from fastapi import HTTPException, status
from app.pipeline.features.osrm_client import osrm_client
from app.services.itinerary_service import ItineraryService


def get_db():
    try:
        conn = db_manager.get_conn()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )

    try:
        yield conn
    finally:
        db_manager.return_conn(conn)

def get_osrm_client():
    return osrm_client


def get_itinerary_service(
    osrm=Depends(get_osrm_client),
) -> ItineraryService:
    return ItineraryService(osrm_client=osrm)