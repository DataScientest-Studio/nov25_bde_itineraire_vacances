"""
Monitoring Prometheus simplifié pour l'API TripMaNGo
"""
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response, Request
from fastapi.responses import PlainTextResponse

# Métriques de l'API
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Active connections to the API'
)

ITINERARY_REQUESTS = Counter(
    'itinerary_requests_total',
    'Total itinerary computation requests',
    ['transport_mode', 'solver', 'days']
)

ITINERARY_DURATION = Histogram(
    'itinerary_computation_duration_seconds',
    'Itinerary computation duration in seconds',
    ['transport_mode', 'solver']
)

ITINERARY_POIS_PROCESSED = Histogram(
    'itinerary_pois_processed',
    'Number of POIs processed per itinerary',
    ['days']
)

def track_itinerary_request(transport_mode: str, solver: str, days: int, pois_count: int, duration: float):
    ITINERARY_REQUESTS.labels(
        transport_mode=transport_mode, 
        solver=solver, 
        days=str(days)
    ).inc()
    
    ITINERARY_DURATION.labels(
        transport_mode=transport_mode, 
        solver=solver
    ).observe(duration)
    
    ITINERARY_POIS_PROCESSED.labels(days=str(days)).observe(pois_count)

async def metrics_endpoint(request: Request = None):
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )