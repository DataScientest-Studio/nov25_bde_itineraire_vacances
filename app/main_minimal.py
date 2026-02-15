from fastapi import FastAPI, Request
from time import time

from app.api.categories import router as categories_router
from app.api.poi import router as poi_router
from app.api.itinerary import router as itinerary_router

from app.monitoring_simple import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    ACTIVE_CONNECTIONS,
    metrics_endpoint,
    track_itinerary_request
)


app = FastAPI()

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    ACTIVE_CONNECTIONS.inc()
    start_time = time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time() - start_time
        ACTIVE_CONNECTIONS.dec()

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()

        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    return response



# Endpoint de monitoring (prioritaire pour éviter les dépendances)
app.add_route("/metrics", metrics_endpoint, methods=["GET"])


@app.get("/")
async def root():
    return {"message": "TripMaNGo API is running"}

# Les routes avec dépendances BDD sont ajoutées après
app.include_router(categories_router)
app.include_router(poi_router)
app.include_router(itinerary_router)