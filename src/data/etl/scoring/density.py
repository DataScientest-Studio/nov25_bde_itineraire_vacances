import polars as pl

#"Nombre de POIs dans le même hexagone"

# Mapping logique → colonne H3
RESOLUTION_MAP = {
    "region": "h3_r6",
    "commune": "h3_r8"
}


def add_density(lf: pl.LazyFrame, level: str = "commune") -> pl.LazyFrame:
    """
    Ajoute une colonne de densité locale basée sur la résolution H3 choisie.

    level : "region" | "commune" | "quartier"
    """

    if level not in RESOLUTION_MAP:
        raise ValueError(f"level doit être {list(RESOLUTION_MAP.keys())}")

    h3_col = RESOLUTION_MAP[level]
    density_col = f"density_{level}"

    # Calcul densité par hexagone
    density = (
        lf.group_by(h3_col)
          .len()
          .rename({"len": density_col})
    )

    # Join sur le LazyFrame original
    return lf.join(density, on=h3_col)