import requests
from typing import List, Tuple, Literal


class OSRMClient:
    def __init__(self, local_url="http://localhost:5000", public_url="https://router.project-osrm.org"):
        self.local_url = local_url.rstrip("/")
        self.public_url = public_url.rstrip("/")
        self.base_url = self._detect_backend()

    def _detect_backend(self):
        try:
            r = requests.get(f"{self.local_url}/health", timeout=0.5)
            if r.status_code == 200:
                return self.local_url
        except:
            pass
        return self.public_url

    @staticmethod
    def _coords_to_str(coords: List[Tuple[float, float]]) -> str:
        return ";".join([f"{lon},{lat}" for lat, lon in coords])

    def table(self, coords, annotations="duration", timeout=10):
        coord_str = self._coords_to_str(coords)
        url = f"{self.base_url}/table/v1/driving/{coord_str}"
        params = {"annotations": annotations}

        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def route(self, start, end, overview="simplified", timeout=10):
        coord_str = self._coords_to_str([start, end])
        url = f"{self.base_url}/route/v1/driving/{coord_str}"
        params = {"overview": overview, "geometries": "geojson"}

        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()["routes"][0]