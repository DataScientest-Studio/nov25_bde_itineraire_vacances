from fastapi import FastAPI
from app.api.categories import router as categories_router
from app.api.poi import router as poi_router
from app.api.itinerary import router as itinerary_router
from app.monitoring_simple import metrics_endpoint

app = FastAPI()

# Endpoint de monitoring (prioritaire pour éviter les dépendances)
app.add_route("/metrics", metrics_endpoint, methods=["GET"])

@app.get("/")
async def root():
    return {"message": "TripMaNGo API is running"}

# Les routes avec dépendances BDD sont ajoutées après
app.include_router(categories_router)
app.include_router(poi_router)
app.include_router(itinerary_router)