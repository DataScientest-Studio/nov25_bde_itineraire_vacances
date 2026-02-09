from app.osrm_client import OSRMClientAsync
from typing import Tuple

class OSRMService:
    def __init__(self, local_url: str, public_url: str):
        self.client = OSRMClientAsync(
            local_url=local_url,
            public_url=public_url,
        )

    async def route_geojson(self, start: Tuple[float, float], end: Tuple[float, float], profile="driving"):
        return await self.client.route_geojson(start, end, profile)
    
    async def route_full(self, coords: list[Tuple[float, float]], profile="driving"):
        return await self.client.route_full(coords, profile)
