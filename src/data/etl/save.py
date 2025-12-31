import polars as pl
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent # parent directory of the script
OUTPUT_DIR = ROOT

def save_parquet(df: pl.DataFrame, output_dir: str = OUTPUT_DIR, versioned: bool = True) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if versioned:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"merged_{ts}.parquet"
    else:
        filename = "merged.parquet"

    output_path = Path(output_dir) / filename
    df.write_parquet(output_path)

    return str(output_path)