from typing import Optional

import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import h3


class SpatialClusterer:
    """
    Regroupe les POIs en clusters spatiaux (1 cluster = 1 jour)
    en utilisant H3 + KMeans, avec prise en compte d'un point d'ancrage.

    Pipeline logique :
        - on travaille au niveau des cellules H3 (h3_8)
        - on regroupe les POIs par cellule et calcule un centroïde (latitude/longitude moyen)
        - on ajoute le point d'ancrage comme "cellule virtuelle"
        - on applique KMeans pour répartir les cellules entre les jours
        - on propage l'étiquette de jour à tous les POIs
    """

    def __init__(self, pois_lf: pl.LazyFrame):
        self.lf = pois_lf
        self.nb_days: int = 1
        self.anchor_latitude: Optional[float] = None
        self.anchor_longitude: Optional[float] = None
        self.h3_resolution: int = 8
        self.random_state: int = 42

    # -----------------------------
    # SET
    # -----------------------------

    def set_nb_days(self, nb_days: int):
        self.nb_days = nb_days
        return self

    def set_anchor(self, latitude: float, longitude: float):
        """
        Définit le point d'ancrage (hôtel, gare, etc.).
        Il sera intégré dans le clustering pour "attirer" les clusters.
        """
        self.anchor_latitude = latitude
        self.anchor_longitude = longitude
        return self

    def set_h3_resolution(self, resolution: int):
        self.h3_resolution = resolution
        return self

    def set_random_state(self, random_state: int):
        self.random_state = random_state
        return self

    # -----------------------------
    # CLUSTERING
    # -----------------------------

    def _build_cells_df(self) -> pl.DataFrame:
        """
        Regroupe les POIs par cellule H3 et calcule un centroïde
        pour chaque cellule (latitude/longitude moyens).
        On matérialise ici car KMeans travaille sur du numpy en mémoire.
        """
        cells_lf = (
            self.lf
            .group_by("h3_r8")
            .agg([
                pl.count().alias("n_pois"),
                pl.mean("latitude").alias("latitude_center"),
                pl.mean("longitude").alias("longitude_center"),
            ])
        )
        cells_df = cells_lf.collect()
        return cells_df

    def _add_anchor_cell(self, cells_df: pl.DataFrame) -> pl.DataFrame:
        """
        Ajoute le point d'ancrage comme cellule H3 virtuelle
        pour influencer le clustering.
        """
        if self.anchor_latitude is None or self.anchor_longitude is None:
            return cells_df

        anchor_h3 = h3.latlng_to_cell(self.anchor_latitude, self.anchor_longitude, self.h3_resolution)

        # Si l'ancrage est déjà dans une cellule existante, on ne duplique pas
        if anchor_h3 in cells_df["h3_r8"].to_list():
            return cells_df

        anchor_row = pl.DataFrame({
            "h3_r8": [anchor_h3],
            "n_pois": [0],  # pas de POI, juste un point d'attraction
            "latitude_center": [self.anchor_latitude],
            "longitude_center": [self.anchor_longitude],
        })

        # harmoniser les types
        anchor_row = anchor_row.cast(cells_df.schema)

        return pl.concat([cells_df, anchor_row], how="vertical")

    def _assign_clusters_to_cells(self, cells_df: pl.DataFrame) -> pl.DataFrame:
        coords = np.vstack(
            [cells_df["latitude_center"].to_numpy(), cells_df["longitude_center"].to_numpy()]
        ).T

        kmeans = KMeans(
            n_clusters=self.nb_days,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = kmeans.fit_predict(coords)

        cells_df = cells_df.with_columns(
            pl.Series("day", labels).cast(pl.Int64)
        )

        return cells_df


    def apply(self) -> pl.LazyFrame:
        """
        Retourne un LazyFrame avec une colongitudene supplémentaire 'day'
        qui indique à quel jour appartient chaque POI (0..nb_days-1).
        """

        # 1. Construction du DF des cellules H3 avec centroïdes
        cells_df = self._build_cells_df()

        # 2. Ajout du point d'ancrage comme cellule virtuelle (optionnel)
        cells_df = self._add_anchor_cell(cells_df)

        # 3. KMeans sur les centroïdes
        cells_df = self._assign_clusters_to_cells(cells_df)

        # 4. Si on a ajouté une cellule virtuelle pour l'ancrage sans POIs,
        cells_df_real = cells_df.filter(pl.col("n_pois") > 0)

        # 5. Join sur le LazyFrame initial pour propager 'day' aux POIs
        cells_lf_with_day = cells_df_real.lazy().select([
            "h3_r8",
            pl.col("day").cast(pl.Int64)
        ])

        lf_with_day = (
            self.lf
            .join(cells_lf_with_day, on="h3_r8", how="left")
        )
        lf_with_day = lf_with_day.with_columns(
            pl.col("day").cast(pl.Int64)
        )

        # Ajout de poi_id (LazyFrame)
        lf_with_day = (
            lf_with_day
            .with_row_index(name="poi_id")              # 0,1,2,3,...
            .with_columns((pl.col("poi_id") + 1))       # 1,2,3,...
        )

        return lf_with_day