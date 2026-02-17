# src/pipelines/023_load_datatourisme_france_prime.py
import os
import hashlib
from typing import List, Dict

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# -----------------------
# CONFIG
# -----------------------
PARQUET_PATH = os.getenv(
    "DATATOURISME_FRANCE_PRIME_PARQUET_PATH",
    "sources/df_datatourisme_france_prime.parquet",
)

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "prime")
DB_USER = os.getenv("POSTGRES_USER", "prime")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "prime")
DSN = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

TARGET_TABLE = os.getenv("DATATOURISME_FRANCE_PRIME_TABLE", "silver.datatourisme_france_prime")


FINAL_COLS: List[str] = [
    "source_id","name","label_en","snippet","snippet_en","lat","lon","country","region_insee","region",
    "dept_insee","departement","city_insee","city","postal_code","address_locality","address","theme_fr","theme_en",
    "media_resource_url","url","rating","review_value_label_fr","review_value_label_en","sub_category","max_people",
    "difficulty_level_fr","locomotion_mode_fr","tour_type_fr","last_update_datatourisme","price",
    "is_label_incontournable","is_label_handicap","is_label_hebergement","is_label_gastronomy","is_label_artisanat",
    "is_label_itinerary","is_label_green","price_level","price_range","main_category","main_cat_weight","type_principal",
    "format_weight","tempo_weight","score_prime","hebergement_type","tour_distance_bucket",
    "practice_duration_final_bucket","seasons_available","opening_hours_str","elevation_gain_loss","source",
    "review_count","distance_km",
]

# Pour l'idempotence, on ignore les placeholders UI (souvent vides au chargement)
HASH_COLS = [c for c in FINAL_COLS if c not in {"review_count", "distance_km"}]

# Types SQL (simple et safe)
COL_TYPES: Dict[str, str] = {
    "source_id": "TEXT",
    "name": "TEXT",
    "label_en": "TEXT",
    "snippet": "TEXT",
    "snippet_en": "TEXT",
    "lat": "DOUBLE PRECISION",
    "lon": "DOUBLE PRECISION",
    "country": "TEXT",
    "region_insee": "INTEGER",
    "region": "TEXT",
    "dept_insee": "INTEGER",
    "departement": "TEXT",
    "city_insee": "INTEGER",
    "city": "TEXT",
    "postal_code": "INTEGER",
    "address_locality": "TEXT",
    "address": "TEXT",
    "theme_fr": "TEXT",
    "theme_en": "TEXT",
    "media_resource_url": "TEXT",
    "url": "TEXT",
    "rating": "DOUBLE PRECISION",
    "review_value_label_fr": "TEXT",
    "review_value_label_en": "TEXT",
    "sub_category": "TEXT",
    "max_people": "INTEGER",
    "difficulty_level_fr": "TEXT",
    "locomotion_mode_fr": "TEXT",
    "tour_type_fr": "TEXT",
    "last_update_datatourisme": "TIMESTAMPTZ",
    "price": "DOUBLE PRECISION",
    "is_label_incontournable": "BOOLEAN",
    "is_label_handicap": "BOOLEAN",
    "is_label_hebergement": "BOOLEAN",
    "is_label_gastronomy": "BOOLEAN",
    "is_label_artisanat": "BOOLEAN",
    "is_label_itinerary": "BOOLEAN",
    "is_label_green": "BOOLEAN",
    "price_level": "TEXT",
    "price_range": "TEXT",
    "main_category": "TEXT",
    "main_cat_weight": "DOUBLE PRECISION",
    "type_principal": "TEXT",
    "format_weight": "DOUBLE PRECISION",
    "tempo_weight": "DOUBLE PRECISION",
    "score_prime": "DOUBLE PRECISION",
    "hebergement_type": "TEXT",
    "tour_distance_bucket": "TEXT",
    "practice_duration_final_bucket": "TEXT",
    "seasons_available": "TEXT",
    "opening_hours_str": "TEXT",
    "elevation_gain_loss": "TEXT",
    "source": "TEXT",
    "review_count": "INTEGER",
    "distance_km": "DOUBLE PRECISION",
    "content_hash": "TEXT",
}


def _stable_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def compute_content_hash(row: pd.Series) -> str:
    s = "|".join(_stable_str(row[c]) for c in HASH_COLS)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_table(conn):
    schema = TARGET_TABLE.split('.')[0]
    table = TARGET_TABLE.split('.')[1]

    cols_sql = ",\n".join([f"{c} {COL_TYPES[c]}" for c in FINAL_COLS] + ["content_hash TEXT"])

    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    DO $$
    BEGIN
        -- supprime type orphelin
        IF EXISTS (
            SELECT 1
            FROM pg_type t
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE t.typname = '{table}' AND n.nspname = '{schema}'
        )
        AND NOT EXISTS (
            SELECT 1
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = '{table}' AND n.nspname = '{schema}'
        )
        THEN
            EXECUTE 'DROP TYPE {schema}.{table} CASCADE';
        END IF;
    END$$;

    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        {cols_sql},
        PRIMARY KEY (source_id)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)



def load_from_df(df: pd.DataFrame):
    # check contrat
    missing = [c for c in FINAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DF incomplet. Colonnes manquantes: {missing}")

    d = df[FINAL_COLS].copy()
    d = d.where(d.notna(), None)

    # hash idempotence
    d["content_hash"] = d.apply(compute_content_hash, axis=1)

    # IMPORTANT: psycopg2 n'accepte pas pd.NA / NAType
    d = d.astype(object).where(pd.notna(d), None)

    cols = FINAL_COLS + ["content_hash"]
    values = [tuple(r) for r in d[cols].to_numpy()]

    set_clause = ",\n        ".join([f"{c} = EXCLUDED.{c}" for c in cols if c != "source_id"])

    sql = f"""
    INSERT INTO {TARGET_TABLE} ({",".join(cols)})
    VALUES %s
    ON CONFLICT (source_id) DO UPDATE
    SET
        {set_clause}
    WHERE {TARGET_TABLE}.content_hash IS DISTINCT FROM EXCLUDED.content_hash;
    """

    with psycopg2.connect(DSN) as conn:
        ensure_table(conn)
        with conn.cursor() as cur:
            execute_values(cur, sql, values, page_size=10000)
        conn.commit()

    print(f"[LOAD DT PRIME] OK rows={len(d):,} -> {TARGET_TABLE}")


def main():
    df = pd.read_parquet(PARQUET_PATH)
    load_from_df(df)
