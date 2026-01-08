import polars as pl

def align_schemas(df_list: list[pl.DataFrame]) -> list[pl.DataFrame]:
    # Récupère toutes les colonnes existantes
    all_cols = set().union(*(df.columns for df in df_list))

    aligned = []
    for df in df_list:
        missing = all_cols - set(df.columns)
        # Ajoute les colonnes manquantes en Null
        for col in missing:
            df = df.with_columns(pl.lit(None).alias(col))
        aligned.append(df.select(sorted(all_cols)))  # ordre stable
    return aligned

def merge_dataframes(df_list: list[pl.DataFrame]) -> pl.DataFrame:
    if not df_list:
        raise ValueError("No dataframes to merge")

    aligned = align_schemas(df_list)
    merged = pl.concat(aligned, how="vertical_relaxed")
    return merged