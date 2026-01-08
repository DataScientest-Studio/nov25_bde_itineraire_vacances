import polars as pl

#"Nombre de POIs dans le même hexagone"

# Mapping logique → colongitudene H3
RESOLUTION_MAP = {
    "region": "h3_r6",
    "commune": "h3_r8"
}


def add_density(lf: pl.LazyFrame, level: str = "commune") -> pl.LazyFrame:
    """
    Ajoute une colongitudene de densité locale basée sur la résolution H3 choisie.

    level : "region" | "commune" | "quartier"
    """

    if level not in RESOLUTION_MAP:
        raise ValueError(f"level doit être {list(RESOLUTION_MAP.keys())}")

    h3_col = RESOLUTION_MAP[level]
    density_col = f"density_{level}"
    density_norm_col = f"{density_col}_norm"

    # 1) Densité brute par hexagone
    density = (
        lf.group_by(h3_col)
        .agg(pl.col("main_category").count().alias(density_col))
    )

    # 2) Normalisation Log
    density = density.with_columns([
        (pl.col(density_col).log1p()).alias("density_log")
    ]).with_columns([
        (
            (pl.col("density_log") - pl.col("density_log").min()) /
            (pl.col("density_log").max() - pl.col("density_log").min())
        ).alias(f"{density_col}_norm")
    ]).drop("density_log")

    # 3) Join sur le LazyFrame original
    return lf.join(density, on=h3_col)
