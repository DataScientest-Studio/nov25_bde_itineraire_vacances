import polars as pl
import h3


def add_h3_columns(
    lf: pl.LazyFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    resolutions = (6, 7, 8, 9)
) -> pl.LazyFrame:
    """
    Ajoute plusieurs colongitudenes H3 (string) à un LazyFrame Polars.
    Exemple : h3_r6, h3_r7, h3_r8, h3_r9

    Compatible avec H3 v4 (latlng_to_cell).
    """

    for res in resolutions:
        lf = lf.with_columns(
            pl.struct([latitude_col, longitude_col]).map_elements(
                lambda row, r=res: (
                    h3.latlng_to_cell(row[latitude_col], row[longitude_col], r)
                    if row[latitude_col] is not None and row[longitude_col] is not None
                    else None
                ),
                return_dtype=pl.String
            ).alias(f"h3_r{res}")
        )

    return lf