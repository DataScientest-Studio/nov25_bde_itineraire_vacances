"""
Exporter Prometheus pour OSRM (Open Source Routing Machine)
"""
import asyncio
import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import httpx

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Métriques OSRM
OSRM_REQUESTS_TOTAL = Counter(
    'osrm_requests_total',
    'Total OSRM requests',
    ['profile', 'endpoint', 'status']
)

OSRM_REQUEST_DURATION = Histogram(
    'osrm_request_duration_seconds',
    'OSRM request duration in seconds',
    ['profile', 'endpoint']
)

OSRM_ACTIVE_PROFILES = Gauge(
    'osrm_active_profiles',
    'Number of active OSRM profiles'
)

OSRM_CACHE_SIZE = Gauge(
    'osrm_cache_size_bytes',
    'OSRM cache size in bytes'
)

app = FastAPI(title="OSRM Exporter", version="1.0.0")

# Configuration des profils OSRM
OSRM_PROFILES = {
    'car': 'http://osrm-car:5000',
    'bike': 'http://osrm-bike:5000', 
    'foot': 'http://osrm-foot:5000'
}

async def check_osrm_health(profile: str, base_url: str) -> Dict[str, Any]:
    """Vérifier la santé d'un profil OSRM"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start_time = time.time()
            
            # Test de l'endpoint de santé
            response = await client.get(f"{base_url}/health")
            duration = time.time() - start_time
            
            if response.status_code == 200:
                OSRM_REQUESTS_TOTAL.labels(
                    profile=profile, 
                    endpoint='health', 
                    status='success'
                ).inc()
                OSRM_REQUEST_DURATION.labels(
                    profile=profile, 
                    endpoint='health'
                ).observe(duration)
                
                return {
                    'profile': profile,
                    'status': 'healthy',
                    'response_time': duration,
                    'base_url': base_url
                }
            else:
                OSRM_REQUESTS_TOTAL.labels(
                    profile=profile, 
                    endpoint='health', 
                    status='error'
                ).inc()
                return {
                    'profile': profile,
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}",
                    'base_url': base_url
                }
                
    except Exception as e:
        OSRM_REQUESTS_TOTAL.labels(
            profile=profile, 
            endpoint='health', 
            status='error'
        ).inc()
        logger.error(f"Error checking {profile}: {e}")
        return {
            'profile': profile,
            'status': 'error',
            'error': str(e),
            'base_url': base_url
        }


@app.get("/health")
async def health():
    """Endpoint de santé de l'exporter"""
    return {"status": "healthy", "service": "osrm-exporter"}

@app.get("/status")
async def status():
    """Statut des profils OSRM"""
    results = []
    
    # Vérifier tous les profils en parallèle
    tasks = [
        check_osrm_health(profile, url) 
        for profile, url in OSRM_PROFILES.items()
    ]
    
    profiles_status = await asyncio.gather(*tasks, return_exceptions=True)
    
    healthy_count = sum(
        1 for status in profiles_status 
        if isinstance(status, dict) and status.get('status') == 'healthy'
    )
    
    # Mettre à jour la métrique des profils actifs
    OSRM_ACTIVE_PROFILES.set(healthy_count)
    
    return {
        "total_profiles": len(OSRM_PROFILES),
        "healthy_profiles": healthy_count,
        "profiles": profiles_status
    }

@app.get("/test/{profile}")
async def test_profile(profile: str):
    """Tester un profil spécifique"""
    if profile not in OSRM_PROFILES:
        return {"error": f"Profile {profile} not found. Available: {list(OSRM_PROFILES.keys())}"}
    
    return await check_osrm_health(profile, OSRM_PROFILES[profile])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
