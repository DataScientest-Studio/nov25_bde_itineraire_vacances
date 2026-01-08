import h3
from h3 import LatLngPoly
from typing import List, Tuple

LatLon = Tuple[float, float]

def latlon_to_h3_str(lat: float, lon: float, res: int) -> str:
    return h3.latlng_to_cell(lat, lon, res)

def polygon_to_cells(polygon: List[LatLon], res: int) -> List[str]:
 
    cells_int = h3.polygon_to_cells(LatLngPoly(polygon), res)
    return [c for c in cells_int]