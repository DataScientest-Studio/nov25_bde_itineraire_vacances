from fastapi import FastAPI
from api.itinerary_router import router as itinerary_router

app = FastAPI(title="Itinerary API")
app.include_router(itinerary_router)