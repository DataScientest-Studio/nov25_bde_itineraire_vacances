from app.core.database import db_manager
from functools import lru_cache
from fastapi import HTTPException, status
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

def get_itinerary_service() -> ItineraryService:
    return ItineraryService(
        osrm_url="http://localhost:5000",
    )
