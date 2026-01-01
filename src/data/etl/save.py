import polars as pl
from pathlib import Path
from datetime import datetime
from .sql.split_tables import split_into_tables

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed" 

# ---------------------------------------------------------
# 1) Sauvegarde Parquet
# ---------------------------------------------------------

def save_parquet(df: pl.DataFrame, output_dir: str = OUTPUT_DIR, versioned: bool = True) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if versioned:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"merged_{ts}.parquet"
    else:
        filename = f"merged.parquet"

    output_path = Path(output_dir) / filename
    df.write_parquet(output_path)

    return str(output_path)

# ---------------------------------------------------------
# 2) Split + Export CSV
# ---------------------------------------------------------
def save_tables_csv(df: pl.DataFrame, output_dir: str = OUTPUT_DIR) -> dict:
    """
    Split le DataFrame enrichi en tables relationnelles
    puis exporte chaque table en CSV dans output_dir.

    Retourne un dict contenant les DataFrames des tables.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Split en tables relationnelles
    tables = split_into_tables(df)

    # Export CSV
    tables["poi"].write_csv(Path(output_dir) / "poi.csv")
    tables["adresse"].write_csv(Path(output_dir) / "adresse.csv")
    tables["main_category"].write_csv(Path(output_dir) / "main_category.csv")
    tables["sub_category"].write_csv(Path(output_dir) / "sub_category.csv")

    return tables
