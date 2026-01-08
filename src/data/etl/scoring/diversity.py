import polars as pl

# Mapping logique → colongitudene H3
RESOLUTION_MAP = {
    "region": "h3_r6",
    "commune": "h3_r8",
}

def add_diversity(lf: pl.LazyFrame, level: str = "commune") -> pl.LazyFrame:
    """
    Ajoute :
    - diversity_{level} : diversité brute (n_unique)
    - diversity_{level}_norm : diversité normalisée (0–1)
    """

    if level not in RESOLUTION_MAP:
        raise ValueError(f"level doit être {list(RESOLUTION_MAP.keys())}")

    h3_col = RESOLUTION_MAP[level]
    diversity_col = f"diversity_{level}"
    diversity_norm_col = f"{diversity_col}_norm"

    # 1) Diversité brute par hexagone
    diversity = (
        lf.group_by(h3_col)
          .agg(pl.col("main_category").n_unique().alias(diversity_col))
    )

    # 2) Normalisation min–max
    diversity = diversity.with_columns([
        (
            (pl.col(diversity_col) - pl.col(diversity_col).min()) /
            (pl.col(diversity_col).max() - pl.col(diversity_col).min())
        ).alias(diversity_norm_col)
    ])

    # 3) Join sur le LazyFrame original
    return lf.join(diversity, on=h3_col)