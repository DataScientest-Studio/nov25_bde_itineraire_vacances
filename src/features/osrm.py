import aiohttp
import asyncio
import math
import numpy as np
from typing import List, Tuple, Literal


class OSRMClientAsync:
    def __init__(
        self,
        local_url="http://localhost:5000",
        public_url="https://router.project-osrm.org",
        max_chunk_size: int = 80,
        max_concurrency: int = 20,   # nombre de requêtes simultanées
    ):
        self.local_url = local_url.rstrip("/")
        self.public_url = public_url.rstrip("/")
        self.base_url = self.public_url  # détection async plus bas
        self.max_chunk_size = max_chunk_size
        self.max_concurrency = max_concurrency

    # async def detect_backend(self):
    #     """Détecte si l’OSRM local est disponible."""
    #     try:
    #         async with aiohttp.ClientSession() as session:
    #             async with session.get(f"{self.local_url}/health", timeout=0.5) as r:
    #                 if r.status == 200:
    #                     self.base_url = self.local_url
    #                     return
    #     except:
    #         pass
    #     self.base_url = self.public_url

    async def detect_backend(self):
        try:
            test_url = f"{self.local_url}/route/v1/driving/2.35,48.85;2.36,48.86"
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, timeout=1) as r:
                    if r.status == 200:
                        self.base_url = self.local_url
                        return
        except:
            pass

        self.base_url = self.public_url

    @staticmethod
    def _coords_to_str(coords: List[Tuple[float, float]]) -> str:
        return ";".join([f"{longitude},{latitude}" for latitude, longitude in coords])

    # ----------------------------------------------------------------------
    # Appel OSRM simple
    # ----------------------------------------------------------------------
    async def _table_raw(self, coords, annotations="duration,distance"):
        coord_str = self._coords_to_str(coords)
        url = f"{self.base_url}/table/v1/driving/{coord_str}"
        params = {"annotations": annotations}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as r:
                r.raise_for_status()
                return await r.json()

    # ----------------------------------------------------------------------
    # Appel OSRM chunké + asynchrone
    # ----------------------------------------------------------------------
    async def table(self, coords, annotations="duration,distance"):
        await self.detect_backend()

        n = len(coords)
        if n == 0:
            raise ValueError("coords est vide")

        # Cas simple : pas besoin de chunk
        if n <= self.max_chunk_size:
            return await self._table_raw(coords, annotations)

        chunk_size = self.max_chunk_size
        num_chunks = math.ceil(n / chunk_size)

        durations = np.zeros((n, n), dtype=float)
        distances = np.zeros((n, n), dtype=float)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_chunk(i, j):
            async with semaphore:
                start_i = i * chunk_size
                end_i = min((i + 1) * chunk_size, n)

                start_j = j * chunk_size
                end_j = min((j + 1) * chunk_size, n)

                sub_coords_src = coords[start_i:end_i]
                sub_coords_dst = coords[start_j:end_j]

                coord_str_src = self._coords_to_str(sub_coords_src)
                coord_str_dst = self._coords_to_str(sub_coords_dst)

                url = f"{self.base_url}/table/v1/driving/{coord_str_src};{coord_str_dst}"

                params = {
                    "sources": ";".join(map(str, range(len(sub_coords_src)))),
                    "destinations": ";".join(
                        map(str, range(len(sub_coords_src), len(sub_coords_src) + len(sub_coords_dst)))
                    ),
                    "annotations": annotations,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as r:
                        r.raise_for_status()
                        data = await r.json()

                return (i, j, data)

        # Lancer toutes les tâches en parallèle
        tasks = [
            process_chunk(i, j)
            for i in range(num_chunks)
            for j in range(num_chunks)
        ]

        for coro in asyncio.as_completed(tasks):
            i, j, data = await coro

            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, n)

            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n)

            if "durations" in data:
                durations[start_i:end_i, start_j:end_j] = data["durations"]

            if "distances" in data:
                distances[start_i:end_i, start_j:end_j] = data["distances"]

        result = {}
        if "duration" in annotations:
            result["durations"] = durations.tolist()
        if "distance" in annotations:
            result["distances"] = distances.tolist()

        return result

    # ----------------------------------------------------------------------
    # Route GeoJSON (async)
    # ----------------------------------------------------------------------
    async def route_geojson(self, start, end):
        await self.detect_backend()

        coord_str = self._coords_to_str([start, end])
        url = f"{self.base_url}/route/v1/driving/{coord_str}"
        params = {"overview": "full", "geometries": "geojson"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as r:
                r.raise_for_status()
                data = await r.json()

        return data["routes"][0]["geometry"]