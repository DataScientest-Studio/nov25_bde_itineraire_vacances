# poi_balancer.py

import polars as pl
from collections import defaultdict

POIS_PER_MODE = {
    "walk": 7,
    "bike": 10,
    "bus": 9,
    "car": 11,
}


class POIBalancer:
    """
    Sélection intelligente des POIs après clustering :
    - diversité main_category / sub_category
    - limite de POIs selon le mode
    - sélection par score
    """

    def __init__(self, lf: pl.LazyFrame):
        self.lf = lf
        self.nb_days = 1
        self.mode = "walk"

    def set_nb_days(self, nb_days: int):
        self.nb_days = nb_days
        return self

    def set_mode(self, mode: str):
        self.mode = mode
        return self

    # ---------------------------------------------------------
    # Sélection par diversité
    # ---------------------------------------------------------

    def _select_diverse_pois(self, df: pl.DataFrame, limit: int) -> pl.DataFrame:
        """
        Sélectionne un ensemble diversifié de POIs :
        - round robin entre catégories
        - tri par score dans chaque catégorie
        """

        # Regrouper par main_category
        groups = defaultdict(list)
        for row in df.iter_rows(named=True):
            groups[row["main_category"]].append(row)

        # Trier chaque catégorie par score
        for cat in groups:
            groups[cat] = sorted(groups[cat], key=lambda r: r["final_score"], reverse=True)

        # Round robin
        selected = []
        cats = list(groups.keys())

        while len(selected) < limit and any(groups.values()):
            for cat in cats:
                if groups[cat]:
                    selected.append(groups[cat].pop(0))
                    if len(selected) == limit:
                        break

        return pl.DataFrame(selected)

    # ---------------------------------------------------------
    # APPLY
    # ---------------------------------------------------------

    def apply(self) -> pl.LazyFrame:
        max_pois = POIS_PER_MODE.get(self.mode, 6)

        dfs = []

        for day in range(self.nb_days):
            df_day = self.lf.filter(pl.col("day") == day).collect()

            if df_day.height == 0:
                continue

            # Diversité + limite
            df_selected = self._select_diverse_pois(df_day, max_pois)

            # Ajouter la colonne day
            df_selected = df_selected.with_columns(
                pl.lit(day).alias("day")
            )

            dfs.append(df_selected)

        if not dfs:
            return pl.DataFrame().lazy()

        
        # 1. Trouver un schéma valide (premier DF non vide)
        schema = None
        for df in dfs:
            if df.height > 0:
                schema = df.schema
                break

        # 2. Si tout est vide → retourner un DF vide
        if schema is None:
            return pl.DataFrame().lazy()

        # 3. Normaliser les DF vides avec un DF vide mais typé
        typed_dfs = []
        for df in dfs:
            if df.height == 0:
                empty_df = pl.DataFrame({
                    col: pl.Series([], dtype=tp)
                    for col, tp in schema.items()
                })
                typed_dfs.append(empty_df)
            else:
                typed_dfs.append(df)


        # 4. Concat propre et garanti sans erreur
        return pl.concat(typed_dfs).lazy()


        #return pl.concat(dfs).lazy()