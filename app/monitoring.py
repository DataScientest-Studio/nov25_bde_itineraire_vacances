"""
Monitoring Prometheus pour l'API TripMaNGo
"""
import time
from functools import wraps
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
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

def track_requests(func):
    """Décorateur pour suivre les requêtes HTTP"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Pour FastAPI, le premier argument est souvent Request
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        try:
            # Incrémenter les connexions actives
            ACTIVE_CONNECTIONS.inc()
            
            # Exécuter la fonction
            result = await func(*args, **kwargs) if hasattr(func, '__call__') and hasattr(func.__call__, '__await__') else func(*args, **kwargs)
            
            # Calculer la durée
            duration = time.time() - start_time
            
            # Extraire les informations de la requête
            if request:
                method = request.method
                endpoint = request.url.path
                status_code = "200"  # Par défaut, à ajuster selon la réponse
                
                # Enregistrer les métriques
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            return result
            
        except Exception as e:
            # En cas d'erreur
            duration = time.time() - start_time
            if request:
                REQUEST_COUNT.labels(
                    method=request.method, 
                    endpoint=request.url.path, 
                    status_code="500"
                ).inc()
                REQUEST_DURATION.labels(
                    method=request.method, 
                    endpoint=request.url.path
                ).observe(duration)
            raise e
        finally:
            # Décrémenter les connexions actives
            ACTIVE_CONNECTIONS.dec()
    
    return wrapper

def track_itinerary_request(transport_mode: str, solver: str, days: int, pois_count: int, duration: float):
    """Enregistrer les métriques de calcul d'itinéraire"""
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

async def metrics_endpoint():
    """Endpoint pour exposer les métriques Prometheus"""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

def get_metrics_summary() -> Dict[str, Any]:
    """Obtenir un résumé des métriques pour le debugging"""
    return {
        "total_requests": REQUEST_COUNT._value._value.sum(),
        "active_connections": ACTIVE_CONNECTIONS._value.get(),
        "avg_request_duration": REQUEST_DURATION.observe.sum() / REQUEST_DURATION.observe.count() if REQUEST_DURATION.observe.count() > 0 else 0,
        "total_itinerary_requests": ITINERARY_REQUESTS._value._value.sum()
    }
