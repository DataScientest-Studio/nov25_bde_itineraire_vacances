import folium
from typing import List
import pandas as pd


def create_route_map(
    pois_df: pd.DataFrame,
    route: List[int],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "id",
    zoom_start: int = 13,
) -> folium.Map:
    """
    Crée une carte Folium avec la route dessinée.
    - pois_df : DataFrame contenant au moins [id_col, lat_col, lon_col]
    - route : liste d'indices (position dans pois_df) ou d'IDs selon ton choix
    """

    # Si route contient des indices de ligne
    coords = pois_df.iloc[route][[lat_col, lon_col]].values

    # Centre sur le premier point
    start_lat, start_lon = coords[0]

    m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom_start)

    # Ajouter les points
    for i, idx in enumerate(route):
        row = pois_df.iloc[idx]
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=f"{i} - {row.get(id_col, idx)}",
            tooltip=f"Stop {i}",
        ).add_to(m)

    # Ajouter la polyline
    folium.PolyLine(
        locations=coords,
        color="blue",
        weight=4,
        opacity=0.7,
    ).add_to(m)

    return m