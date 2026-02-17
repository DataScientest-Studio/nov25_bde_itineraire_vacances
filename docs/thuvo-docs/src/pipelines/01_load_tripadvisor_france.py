# src/pipelines/01_load_tripadvisor_france.py
import os
import json
from typing import Any, List

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# ============================================================
# Config
# ============================================================
PARQUET_PATH = os.getenv("TRIPADVISOR_PARQUET_PATH", "sources/df_tripadvisor_france.parquet")

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "prime")
DB_USER = os.getenv("POSTGRES_USER", "prime")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "prime")
DSN = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

# RAW target (IMPORTANT: schema-qualified)
RAW_TABLE = os.getenv("TRIPADVISOR_RAW_TABLE", "raw.tripadvisor_france")


# Ordre IMPORTANT = contrat d'insertion
RAW_COLS: List[str] = [
    "source_id", "source", "type", "url", "lat", "lon",
    "name", "address", "postal_code", "city", "region", "country",
    "snippet", "rating", "review_count",
    "price_level", "price_range", "price",
    "reco_score",
    "cuisine_continent", "cuisine_type",
    "vegetarian_friendly", "vegan_options", "gluten_free",
    "is_halal", "is_kosher",
    "awards_binary",
    "max_people", "distance_km",
    # meta RAW
    "raw_loaded_at", "raw_source_file", "ingested_at", "updated_at",
]


# ============================================================
# Utils
# ============================================================
def to_sql_value(x: Any):
    """Convert pandas/numpy values to psycopg2-friendly values."""
    if x is None:
        return None
    try:
        if pd.isna(x) is True:
            return None
    except Exception:
        pass

    if isinstance(x, (np.generic,)):
        return x.item()

    if isinstance(x, (np.ndarray, list, tuple, set)):
        try:
            return json.dumps(list(x), ensure_ascii=False)
        except Exception:
            return str(list(x))

    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)

    return x


def ensure_contract(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Ensure required columns exist + basic typing/cleaning aligned with RAW."""
    BOOL_COLS = {
        "vegetarian_friendly", "vegan_options", "gluten_free",
        "is_halal", "is_kosher", "awards_binary"
    }

    # Add missing cols from drift
    base_cols = [c for c in RAW_COLS if c not in {"raw_loaded_at", "raw_source_file", "ingested_at", "updated_at"}]
    for c in base_cols:
        if c not in df.columns:
            df[c] = False if c in BOOL_COLS else pd.NA

    # Keep only relevant columns (without meta first)
    keep = [
        "source_id", "source", "type", "url", "lat", "lon",
        "name", "address", "postal_code", "city", "region", "country",
        "snippet", "rating", "review_count",
        "price_level", "price_range", "price",
        "reco_score",
        "cuisine_continent", "cuisine_type",
        "vegetarian_friendly", "vegan_options", "gluten_free",
        "is_halal", "is_kosher",
        "awards_binary",
        "max_people", "distance_km",
    ]
    df = df[keep].copy()

    # Types numeric
    for c in ["lat", "lon", "rating", "review_count", "price", "reco_score", "distance_km", "max_people"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Booleans
    for c in BOOL_COLS:
        df[c] = df[c].fillna(False).astype(bool)

    # Strings (snippet sometimes huge -> keep as text)
    str_cols = [
        "source_id", "source", "type", "url",
        "name", "address", "postal_code", "city", "region", "country",
        "snippet", "price_level", "price_range",
        "cuisine_continent", "cuisine_type",
    ]
    for c in str_cols:
        df[c] = df[c].astype("string")

    df["source_id"] = df["source_id"].fillna("").astype("string")
    df = df[df["source_id"].str.strip() != ""]  # drop empty IDs

    # snippet might be non-string (list/dict) -> stringify
    df["snippet"] = df["snippet"].apply(to_sql_value).astype("string").fillna("")

    # Meta columns (set in python so they always exist)
    now = pd.Timestamp.utcnow()
    df["raw_loaded_at"] = now
    df["ingested_at"] = now
    df["updated_at"] = now
    df["raw_source_file"] = source_file

    # Reorder exactly as RAW_COLS
    df = df[RAW_COLS].copy()
    return df


def _to_rows(df: pd.DataFrame, cols: List[str]) -> List[tuple]:
    df2 = df[cols].copy()
    for c in cols:
        df2[c] = df2[c].map(to_sql_value)
    df2 = df2.astype(object).where(pd.notna(df2), None)
    return [tuple(r) for r in df2.to_numpy()]


def _ensure_raw_table(cur) -> None:
    """Safety net: ensure schema exists + required meta cols exist + unique constraint for upsert."""
    cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    # Ensure meta columns exist (in case table was created earlier without them)
    cur.execute(f"ALTER TABLE {RAW_TABLE} ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ DEFAULT now();")
    cur.execute(f"ALTER TABLE {RAW_TABLE} ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ DEFAULT now();")
    cur.execute(f"ALTER TABLE {RAW_TABLE} ADD COLUMN IF NOT EXISTS raw_loaded_at TIMESTAMPTZ DEFAULT now();")
    cur.execute(f"ALTER TABLE {RAW_TABLE} ADD COLUMN IF NOT EXISTS raw_source_file TEXT;")

    # For idempotent UPSERT we need a unique constraint/index on source_id
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_tripadvisor_source_id ON {RAW_TABLE}(source_id);")


# ============================================================
# Load
# ============================================================
def load_raw_from_df(df: pd.DataFrame) -> None:
    source_file = os.path.basename(PARQUET_PATH)
    df = ensure_contract(df, source_file=source_file)

    rows = _to_rows(df, RAW_COLS)

    insert_cols_sql = ", ".join(RAW_COLS)

    # On conflict: update everything except source_id, keep raw_loaded_at as "load time" (refresh),
    # and updated_at as refresh time too.
    update_cols = [c for c in RAW_COLS if c != "source_id"]
    update_set_sql = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])

    sql = f"""
        INSERT INTO {RAW_TABLE} ({insert_cols_sql})
        VALUES %s
        ON CONFLICT (source_id) DO UPDATE
        SET {update_set_sql};
    """

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            _ensure_raw_table(cur)
            execute_values(cur, sql, rows, page_size=5000)
        conn.commit()

    print(f"[TripAdvisor RAW] upserted_rows={len(df):,} into {RAW_TABLE} (source_file={source_file})")


def main() -> None:
    df = pd.read_parquet(PARQUET_PATH)
    load_raw_from_df(df)


if __name__ == "__main__":
    main()
