from fastapi import FastAPI, Depends
from functools import lru_cache
from pathlib import Path

from api.database import DBManager
import api.database as db
from api.models import (
    CategoriesRequest,
    ItineraryRequest,

    ItineraryResponse
)
from services.itinerary_service import ItineraryService


app = FastAPI(title="Itinerary API")

# ---------------------------------------------------------
#  DATABASE DEPENDENCY
# ---------------------------------------------------------

@lru_cache
def get_db():
    return db.DBManager(
    )

@app.get("/main_categories")
def get_main_categories(dbm: db.DBManager = Depends(get_db)):
    return {"main_categories": dbm.get_main_categories()}


@app.post("/sub_categories")
def get_sub_categories(req: CategoriesRequest, dbm: db.DBManager = Depends(get_db)):
    sub_categories = dbm.get_sub_categories(req.categories_list)
    return {"sub_categories": sub_categories}


# ---------------------------------------------------------
#  ITINERARY PIPELINE DEPENDENCY
# ---------------------------------------------------------

@lru_cache
def get_service():
    return ItineraryService(
        osrm_url="http://localhost:5000"
    )


@app.post("/itinerary", response_model=ItineraryResponse)
def compute_itinerary(
    req: ItineraryRequest,
    dbm: DBManager = Depends(get_db),
    service: ItineraryService = Depends(get_service)
):

    pois_df = dbm.get_pois_filtered(
        main_categories=req.main_categories,
        sub_categories=req.sub_categories,
        lat=req.start.lat,
        lon=req.start.lon,
        radius=req.radius,
        min_score=req.min_score,
        nb_days=req.nb_days,
        transport_mode=req.transport_mode,
        solver=req.solver
    )

    itinerary = service.compute(req, pois_df)

    return itinerary
