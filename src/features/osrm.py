import asyncio
import math
from typing import List, Tuple

import aiohttp
import numpy as np


class OSRMClientAsync:
    """
    Client OSRM asynchrone avec :
    - détection backend local/public
    - support du profil (foot, bike, driving)
    - chunking pour grandes matrices
    """

    def __init__(
        self,
        local_url="http://localhost:5000",
        public_url="https://router.project-osrm.org",
        max_chunk_size: int = 80,
        max_concurrency: int = 20,
    ):
        self.local_url = local_url.rstrip("/")
        self.public_url = public_url.rstrip("/")
        self.base_url = self.public_url
        self.max_chunk_size = max_chunk_size
        self.max_concurrency = max_concurrency

    # ---------------------------------------------------------
    # Détection backend
    # ---------------------------------------------------------
    async def detect_backend(self):
        """
        Vérifie si l’OSRM local est disponible.
        Sinon bascule sur OSRM public.
        """
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

    # ---------------------------------------------------------
    # Utilitaire : conversion coords → "lon,lat;lon,lat"
    # ---------------------------------------------------------
    @staticmethod
    def _coords_to_str(coords: List[Tuple[float, float]]) -> str:
        # coords = [(lon, lat), ...]
        return ";".join([f"{lon},{lat}" for lon, lat in coords])

    # ---------------------------------------------------------
    # Appel OSRM simple (sans chunk)
    # ---------------------------------------------------------
    async def _table_raw(self, coords, annotations="duration,distance", profile="foot"):
        coord_str = self._coords_to_str(coords)
        url = f"{self.base_url}/table/v1/{profile}/{coord_str}"
        params = {"annotations": annotations}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as r:
                r.raise_for_status()
                return await r.json()

    # ---------------------------------------------------------
    # Appel OSRM chunké + asynchrone
    # ---------------------------------------------------------
    async def table(self, coords, annotations="duration,distance", profile="foot"):
        """
        coords = [(lon, lat), ...]
        profile = "foot" | "bike" | "driving"
        """
        await self.detect_backend()

        n = len(coords)
        if n == 0:
            raise ValueError("coords est vide")

        # Cas simple : pas besoin de chunk
        if n <= self.max_chunk_size:
            return await self._table_raw(coords, annotations, profile)

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

                url = f"{self.base_url}/table/v1/{profile}/{coord_str_src};{coord_str_dst}"

                params = {
                    "sources": ";".join(map(str, range(len(sub_coords_src)))),
                    "destinations": ";".join(
                        map(
                            str,
                            range(
                                len(sub_coords_src),
                                len(sub_coords_src) + len(sub_coords_dst),
                            ),
                        )
                    ),
                    "annotations": annotations,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as r:
                        r.raise_for_status()
                        data = await r.json()

                return (i, j, data)

        # Lancer toutes les tâches
        tasks = [
            process_chunk(i, j) for i in range(num_chunks) for j in range(num_chunks)
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

    # ---------------------------------------------------------
    # Route GeoJSON
    # ---------------------------------------------------------
    async def route_geojson(self, start, end, profile="driving"):
        """
        start/end = (lon, lat)
        """
        await self.detect_backend()

        coord_str = self._coords_to_str([start, end])
        url = f"{self.base_url}/route/v1/{profile}/{coord_str}"
        params = {"overview": "full", "geometries": "geojson"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as r:
                r.raise_for_status()
                data = await r.json()

        return data["routes"][0]["geometry"]
