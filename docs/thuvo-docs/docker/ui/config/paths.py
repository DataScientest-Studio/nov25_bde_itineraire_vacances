# ================================================
# PATHS ---> docker/ui/config/paths.py
# ================================================

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    base_dir: str
    dt_path: str
    ta_path: str
    ab_path: str


def get_base_dir() -> str:
    """
    Ordre de résolution:
    1) ENV ITIVAC_BASE_DIR (recommandé en Docker/prod)
    2) /data
    3) /app/sources
    4) fallback local dynamique: <repo>/sources (pas de chemin Windows hardcodé)
    """
    # 1) variable d'env prioritaire
    env_base = os.getenv("ITIVAC_BASE_DIR")
    if env_base:
        return env_base

    # 2) option docker: volume monté dans /data
    if Path("/data").is_dir():
        return "/data"

    # 3) ton cas docker actuel: repo monté en /app
    app_sources = Path("/app/sources")
    if app_sources.is_dir():
        return str(app_sources)

    # 4) fallback local (dynamique)
    # paths.py = .../docker/ui/config/paths.py -> remonter jusqu'à la racine repo puis /sources
    repo_root = Path(__file__).resolve().parents[3]  # .../docker
    # mais ici on cherche "sources" de façon robuste
    candidates = [
        repo_root / "sources",
        repo_root.parent / "sources",
        Path.cwd() / "sources",
    ]
    for c in candidates:
        if c.is_dir():
            return str(c)

    # dernier recours: cwd (ça évite un chemin Windows)
    return str(Path.cwd())


def get_data_paths(base_dir: str | None = None) -> DataPaths:
    base = base_dir or get_base_dir()

    return DataPaths(
        base_dir=base,
        dt_path=os.path.join(base, "df_prime_classique.parquet"),
        ta_path=os.path.join(base, "df_tripadvisor_France.parquet"),
        ab_path=os.path.join(base, "df_airbnb_Paris.parquet"),
    )
