import time
import polars as pl

class POISelector:
    """
    Sélection intelligente de POIs par cluster (jour),
    avec :
      - score mixte (final_score + diversité subcat)
      - profiling intégré
      - >= 2 restaurants
      - diversité par sub_categorie
      - limite globale dépendante du mode de transport
    """

    def __init__(
        self,
        transport_mode: str,
        min_restaurants_per_cluster: int = 2,
        max_pois_per_subcategorie: int = 3,
        restaurant_label: str = "Gastronomie & Restauration",
        w_final_score: float = 0.7,
        w_diversity: float = 0.3,
        diversity_col: str = "diversity_subcat_norm",
    ):
        # paramètres
        self.transport_mode = transport_mode.lower()
        self.min_restaurants = min_restaurants_per_cluster
        self.max_per_subcat = max_pois_per_subcategorie
        self.restaurant_label = restaurant_label

        # score mixte
        self.w_final = w_final_score
        self.w_div = w_diversity
        self.diversity_col = diversity_col

        # profiling
        self.profiling = {}

        # limite globale
        self.max_pois_per_cluster = self._max_pois_from_transport_mode()

    # ------------------------------------------------------------------
    # Diversité sub_category
    # ------------------------------------------------------------------
    def _compute_diversity_subcat_norm(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .with_columns([
                pl.count().over(["cluster_id", "sub_category"]).alias("subcat_count")
            ])
            .with_columns([
                (1 / (1 + pl.col("subcat_count"))).alias("diversity_raw")
            ])
            .with_columns([
                (
                    (pl.col("diversity_raw") - pl.col("diversity_raw").min().over("cluster_id"))
                    / (pl.col("diversity_raw").max().over("cluster_id") - pl.col("diversity_raw").min().over("cluster_id"))
                ).alias("diversity_subcat_norm")
            ])
            .drop(["subcat_count", "diversity_raw"])
        )

    # ------------------------------------------------------------------
    # Profiling helper
    # ------------------------------------------------------------------
    def _profile(self, name: str, start_time: float):
        duration = time.time() - start_time
        self.profiling[name] = self.profiling.get(name, 0) + duration

    # ------------------------------------------------------------------
    # 1) Déterminer le nb max de POIs selon le mode de transport
    # ------------------------------------------------------------------
    def _max_pois_from_transport_mode(self) -> int:
        if self.transport_mode == "walk":
            return 14
        if self.transport_mode == "bike":
            return 22
        if self.transport_mode == "bus":
            return 24
        if self.transport_mode == "car":
            return 28
        raise ValueError(f"Unknown transport mode: {self.transport_mode}")

    # ------------------------------------------------------------------
    # 2) Ajouter le score mixte
    # ------------------------------------------------------------------
    def _add_mixed_score(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        start = time.time()

        lf2 = lf.with_columns(
            (
                self.w_final * pl.col("final_score")
                + self.w_div * pl.col(self.diversity_col)
            ).alias("mixed_score")
        )

        self._profile("add_mixed_score", start)
        return lf2

    # ------------------------------------------------------------------
    # 3) Sélection des restaurants
    # ------------------------------------------------------------------
    def _select_restaurants(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        start = time.time()

        lf2 = (
            lf
            .filter(pl.col("main_category") == self.restaurant_label)
            .with_columns(
                pl.col("mixed_score")
                .rank("dense", descending=True)
                .over("cluster_id")
                .alias("rank_restaurant")
            )
            .filter(pl.col("rank_restaurant") <= self.min_restaurants)
            .drop("rank_restaurant")
        )

        self._profile("select_restaurants", start)
        return lf2

    # ------------------------------------------------------------------
    # 4) Sélection des POIs diversifiés par sub_categorie
    # ------------------------------------------------------------------
    def _select_diverse_pois(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        start = time.time()

        lf2 = (
            lf
            .filter(pl.col("main_category") != self.restaurant_label)
            .with_columns(
                pl.col("mixed_score")
                .rank("dense", descending=True)
                .over(["cluster_id", "sub_category"])
                .alias("rank_subcat")
            )
            .filter(pl.col("rank_subcat") <= self.max_per_subcat)
            .drop("rank_subcat")
        )

        self._profile("select_diverse_pois", start)
        return lf2

    # ------------------------------------------------------------------
    # 5) Fusion + limite globale par cluster
    # ------------------------------------------------------------------
    def _combine_and_limit(
        self,
        lf_restos: pl.LazyFrame,
        lf_non_restos: pl.LazyFrame,
    ) -> pl.LazyFrame:
        # 1) Restaurants protégés
        lf_restos = lf_restos.with_columns(pl.lit(True).alias("is_restaurant"))

        # 2) Non-restos
        lf_non_restos = lf_non_restos.with_columns(pl.lit(False).alias("is_restaurant"))

        # 3) Fusion
        lf_all = pl.concat([lf_restos, lf_non_restos]).unique(
            subset=["cluster_id", "poi_id"]
        )

        # 4) Séparer restaurants et non-restaurants
        lf_r = lf_all.filter(pl.col("is_restaurant"))
        lf_nr = lf_all.filter(~pl.col("is_restaurant"))

        # 5) Limite globale appliquée uniquement aux non-restos
        lf_nr_limited = (
            lf_nr
            .with_columns(
                pl.col("mixed_score")
                .rank("dense", descending=True)
                .over("cluster_id")
                .alias("rank_cluster")
            )
            .filter(pl.col("rank_cluster") <= self.max_pois_per_cluster - self.min_restaurants)
            .drop("rank_cluster")
        )

        # 6) Fusion finale
        return pl.concat([lf_r, lf_nr_limited]).drop("is_restaurant")
      

    # ------------------------------------------------------------------
    # 6) Pipeline principal
    # ------------------------------------------------------------------
    def select(self, lf_pois: pl.LazyFrame) -> pl.LazyFrame:
        """
        Retourne un LazyFrame final prêt pour OSRM :
          - cluster_id
          - poi_id
          - lat, lon
          - main_category
          - sub_category
          - final_score
          - mixed_score
        """
        # 1) Calcul diversité subcat
        lf_pois = self._compute_diversity_subcat_norm(lf_pois)

        # 2) Score mixte
        lf_scored = self._add_mixed_score(lf_pois)

        # 3) restaurants
        lf_restos = self._select_restaurants(lf_scored)

        # 4) diversité
        lf_non_restos = self._select_diverse_pois(lf_scored)

        # 5) limite globale
        lf_selected = self._combine_and_limit(lf_restos, lf_non_restos)

        # 6) tri final
        start = time.time()
        lf_final = (
            lf_selected
            .sort([
                "cluster_id",
                pl.col("main_category"),
                "sub_category",
                pl.col("mixed_score").sort(descending=True),
            ])
            .select([
                "cluster_id",
                "poi_id",
                "latitude",
                "longitude",
                "main_category",
                "sub_category",
                "final_score",
                "mixed_score",
            ])
        )
        self._profile("final_sort_and_select", start)

        return lf_final