import polars as pl

# Diversité des catégories dans l’hexagone

# Mapping logique → colonne H3
RESOLUTION_MAP = {
    "region": "h3_r6",
    "commune": "h3_r8"
}


def add_diversity(lf: pl.LazyFrame, level: str = "commune") -> pl.LazyFrame:
    """
    Ajoute une colonne de diversité locale basée sur la résolution H3 choisie.

    level : "region" | "commune" | "quartier"
    """

    if level not in RESOLUTION_MAP:
        raise ValueError(f"level doit être {list(RESOLUTION_MAP.keys())}")

    h3_col = RESOLUTION_MAP[level]
    diversity_col = f"diversity_{level}"

    # Calcul diversité par hexagone
    diversity = (
        lf.group_by(h3_col)
          .agg(pl.col("main_category").n_unique().alias(diversity_col))
    )

    # Join sur le LazyFrame original
    return lf.join(diversity, on=h3_col)