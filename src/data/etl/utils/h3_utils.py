import h3
from h3 import latitudeLngPoly
from typing import List, Tuple

latitudelongitude = Tuple[float, float]

def latitudelongitude_to_h3_str(latitude: float, longitude: float, res: int) -> str:
    return h3.latlng_to_cell(latitude, longitude, res)

def polygon_to_cells(polygon: List[latitudelongitude], res: int) -> List[str]:
 
    cells_int = h3.polygon_to_cells(latitudeLngPoly(polygon), res)
    return [c for c in cells_int]