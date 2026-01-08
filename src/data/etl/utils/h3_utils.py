import h3
from h3 import LatLngPoly
from typing import List, Tuple

LatLon = Tuple[float, float]

<<<<<<< HEAD
def latlon_to_h3_str(latitude: float, longitude: float, res: int) -> str:
    return h3.latlng_to_cell(latitude, longitude, res)
=======
def latlon_to_h3_str(lat: float, lon: float, res: int) -> str:
    return h3.latlng_to_cell(lat, lon, res)
>>>>>>> 2c202210cd102230a91472e461a9227c9eeb0121

def polygon_to_cells(polygon: List[LatLon], res: int) -> List[str]:
 
    cells_int = h3.polygon_to_cells(LatLngPoly(polygon), res)
    return [c for c in cells_int]