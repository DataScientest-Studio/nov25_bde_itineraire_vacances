from pathlib import Path
from typing import Dict

import polars as pl


def save_benchmark(results: Dict[str, pl.DataFrame], folder: str = "benchmark_results"):
    """
    Sauvegarde chaque DataFrame du benchmark dans un dossier.
    Format : Parquet (rapide, compact, parfait pour Polars)
    """
    path = Path(folder)
    path.mkdir(exist_ok=True)

    for name, df in results.items():
        file_path = path / f"{name}.parquet"
        df.write_parquet(file_path)
        print(f"✓ Sauvegardé : {file_path}")


def load_benchmark(folder: str = "benchmark_results") -> Dict[str, pl.DataFrame]:
    """
    Recharge les résultats du benchmark depuis un dossier.
    """
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    results = {}
    for file in path.glob("*.parquet"):
        key = file.stem
        results[key] = pl.read_parquet(file)
        print(f"✓ Chargé : {file}")

    return results
