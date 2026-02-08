from .osrm import OSRMClientAsync

# Instance unique (singleton)
osrm_client = OSRMClientAsync(
    local_url="http://localhost",      
    public_url="https://router.project-osrm.org",
    max_chunk_size=80,
    max_concurrency=20,
)