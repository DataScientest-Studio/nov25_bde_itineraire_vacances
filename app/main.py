from fastapi import FastAPI
from app.api.categories import router as categories_router
from app.api.poi import router as poi_router
from app.api.itinerary import router as itinerary_router

app = FastAPI()

app.include_router(categories_router)
app.include_router(poi_router)
app.include_router(itinerary_router)