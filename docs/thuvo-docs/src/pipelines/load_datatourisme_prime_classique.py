import hashlib

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

PARQUET_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances3\df_prime_classique.parquet"
DSN = "host=localhost port=5432 dbname=itineraire_vacances_prime user=thuvo password=admin"

TABLE = "silver.prime_classique"
HIST_TABLE = "silver.prime_classique_history"
STG = "_stg_prime_classique"

# Colonnes dans l'ordre logique (mêmes noms que parquet)
BASE_COLS = [
    "source_id",
    "poi_types",
    "name",
    "label_en",
    "snippet",
    "snippet_en",
    "lat",
    "lon",
    "country",
    "region_insee",
    "region",
    "dept_insee",
    "departement",
    "city_insee",
    "city",
    "postal_code",
    "address_locality",
    "address",
    "theme_fr",
    "theme_en",
    "architectural_style_fr",
    "architectural_style_en",
    "main_media_url",
    "media_resource_url",
    "url",
    "rating",
    "review_value_label_fr",
    "review_value_label_en",
    "sub_category",
    "opens_time",
    "closes_time",
    "hours_valid_from",
    "hours_valid_through",
    "max_people",
    "difficulty_level_fr",
    "locomotion_mode_fr",
    "tour_type_fr",
    "tour_distance_m",
    "duration_min",
    "practice_duration_min",
    "duration_days",
    "practice_duration_days",
    "positive_elevation_gain_m",
    "negative_elevation_loss_m",
    "start_date",
    "end_date",
    "last_update_datatourisme",
    "price",
    "min_price",
    "max_price",
    "type_principal",
    "is_resto_type",
    "is_resto_label",
    "is_resto",
    "is_label_incontournable",
    "is_label_famille",
    "is_label_handicap",
    "is_label_hebergement",
    "is_label_gastronomie",
    "is_label_artisanat",
    "is_label_randonnee",
    "is_label_green",
    "is_etoile",
    "price_level",
    "main_category_compressed",
    "is_prime_plus",
    "main_cat_weight",
    "main_cat_candidates",
    "format_label",
    "format_weight",
    "tempo_label",
    "tempo_weight",
    "score_prime",
    "source",
    "review_count",
    "distance_km",
]

# Colonnes qui sont "numériques" en cible, mais qu'on accepte en TEXT en staging (pour éviter CEDEX etc.)
CAST_INT_COLS = {
    "region_insee",
    "dept_insee",
    "city_insee",
    "max_people",
    "review_count",
}


def stable_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def compute_hash_row(row):
    s = "|".join(stable_str(row[c]) for c in BASE_COLS)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    df = pd.read_parquet(PARQUET_PATH)

    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le parquet: {missing}")

    # Casts safe / cohérents
    df["source_id"] = df["source_id"].astype(str)

    # Champs pouvant contenir CEDEX / valeurs non numériques -> staging TEXT
    df["postal_code"] = df["postal_code"].astype("string")
    for c in CAST_INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # Certains champs object/list -> stringify
    for c in [
        "poi_types",
        "theme_fr",
        "theme_en",
        "type_principal",
        "source",
        "main_cat_candidates",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # Hash
    df["content_hash"] = df.apply(compute_hash_row, axis=1)

    # NA -> None (psycopg2 compatible)
    df_to_load = df[BASE_COLS + ["content_hash"]].copy()
    df_to_load = df_to_load.astype(object).where(pd.notna(df_to_load), None)
    rows = [tuple(r) for r in df_to_load.to_numpy()]

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            # 1) staging temp (permissif)
            cur.execute(
                f"""
                DROP TABLE IF EXISTS {STG};
                CREATE TEMP TABLE {STG} (
                  source_id TEXT,
                  poi_types TEXT,
                  name TEXT,
                  label_en TEXT,
                  snippet TEXT,
                  snippet_en TEXT,
                  lat DOUBLE PRECISION,
                  lon DOUBLE PRECISION,
                  country TEXT,

                  region_insee TEXT,
                  region TEXT,
                  dept_insee TEXT,
                  departement TEXT,
                  city_insee TEXT,
                  city TEXT,
                  postal_code TEXT,

                  address_locality TEXT,
                  address TEXT,

                  theme_fr TEXT,
                  theme_en TEXT,
                  architectural_style_fr TEXT,
                  architectural_style_en TEXT,

                  main_media_url TEXT,
                  media_resource_url TEXT,
                  url TEXT,

                  rating DOUBLE PRECISION,
                  review_value_label_fr TEXT,
                  review_value_label_en TEXT,
                  sub_category TEXT,

                  opens_time TEXT,
                  closes_time TEXT,
                  hours_valid_from TEXT,
                  hours_valid_through TEXT,

                  max_people TEXT,

                  difficulty_level_fr TEXT,
                  locomotion_mode_fr TEXT,
                  tour_type_fr TEXT,

                  tour_distance_m DOUBLE PRECISION,
                  duration_min DOUBLE PRECISION,
                  practice_duration_min DOUBLE PRECISION,
                  duration_days DOUBLE PRECISION,
                  practice_duration_days DOUBLE PRECISION,
                  positive_elevation_gain_m DOUBLE PRECISION,
                  negative_elevation_loss_m DOUBLE PRECISION,

                  start_date TEXT,
                  end_date TEXT,
                  last_update_datatourisme TEXT,

                  price DOUBLE PRECISION,
                  min_price DOUBLE PRECISION,
                  max_price DOUBLE PRECISION,

                  type_principal TEXT,

                  is_resto_type BOOLEAN,
                  is_resto_label BOOLEAN,
                  is_resto BOOLEAN,

                  is_label_incontournable BOOLEAN,
                  is_label_famille BOOLEAN,
                  is_label_handicap BOOLEAN,
                  is_label_hebergement BOOLEAN,
                  is_label_gastronomie BOOLEAN,
                  is_label_artisanat BOOLEAN,
                  is_label_randonnee BOOLEAN,
                  is_label_green BOOLEAN,
                  is_etoile BOOLEAN,

                  price_level TEXT,
                  main_category_compressed TEXT,
                  is_prime_plus BOOLEAN,

                  main_cat_weight DOUBLE PRECISION,
                  main_cat_candidates TEXT,
                  format_label TEXT,
                  format_weight DOUBLE PRECISION,
                  tempo_label TEXT,
                  tempo_weight DOUBLE PRECISION,

                  score_prime DOUBLE PRECISION,
                  source TEXT,

                  review_count TEXT,
                  distance_km DOUBLE PRECISION,

                  content_hash TEXT
                ) ON COMMIT DROP;
            """
            )

            insert_sql = f"""
                INSERT INTO {STG} (
                  {", ".join(BASE_COLS)}, content_hash
                ) VALUES %s
            """
            execute_values(cur, insert_sql, rows, page_size=5000)

            # Expressions de cast safe vers INTEGER (sinon NULL)
            cast_region_insee = (
                "CASE WHEN s.region_insee ~ '^\\d+$' THEN s.region_insee::int END"
            )
            cast_dept_insee = (
                "CASE WHEN s.dept_insee   ~ '^\\d+$' THEN s.dept_insee::int END"
            )
            cast_city_insee = (
                "CASE WHEN s.city_insee   ~ '^\\d+$' THEN s.city_insee::int END"
            )
            cast_max_people = (
                "CASE WHEN s.max_people   ~ '^\\d+$' THEN s.max_people::int END"
            )
            cast_review_count = (
                "CASE WHEN s.review_count ~ '^\\d+$' THEN s.review_count::int END"
            )

            # 2) insert nouveaux (vers table finale typée)
            cur.execute(
                f"""
                INSERT INTO {TABLE} (
                  {", ".join(BASE_COLS)},
                  geom, content_hash, updated_at
                )
                SELECT
                  s.source_id,
                  s.poi_types,
                  s.name,
                  s.label_en,
                  s.snippet,
                  s.snippet_en,
                  s.lat,
                  s.lon,
                  s.country,

                  {cast_region_insee},
                  s.region,
                  {cast_dept_insee},
                  s.departement,
                  {cast_city_insee},
                  s.city,
                  s.postal_code,

                  s.address_locality,
                  s.address,

                  s.theme_fr,
                  s.theme_en,
                  s.architectural_style_fr,
                  s.architectural_style_en,

                  s.main_media_url,
                  s.media_resource_url,
                  s.url,

                  s.rating,
                  s.review_value_label_fr,
                  s.review_value_label_en,
                  s.sub_category,

                  s.opens_time,
                  s.closes_time,
                  s.hours_valid_from,
                  s.hours_valid_through,

                  {cast_max_people},

                  s.difficulty_level_fr,
                  s.locomotion_mode_fr,
                  s.tour_type_fr,
                  s.tour_distance_m,
                  s.duration_min,
                  s.practice_duration_min,
                  s.duration_days,
                  s.practice_duration_days,
                  s.positive_elevation_gain_m,
                  s.negative_elevation_loss_m,

                  s.start_date,
                  s.end_date,
                  s.last_update_datatourisme,

                  s.price,
                  s.min_price,
                  s.max_price,

                  s.type_principal,

                  s.is_resto_type,
                  s.is_resto_label,
                  s.is_resto,

                  s.is_label_incontournable,
                  s.is_label_famille,
                  s.is_label_handicap,
                  s.is_label_hebergement,
                  s.is_label_gastronomie,
                  s.is_label_artisanat,
                  s.is_label_randonnee,
                  s.is_label_green,
                  s.is_etoile,

                  s.price_level,
                  s.main_category_compressed,
                  s.is_prime_plus,

                  s.main_cat_weight,
                  s.main_cat_candidates,
                  s.format_label,
                  s.format_weight,
                  s.tempo_label,
                  s.tempo_weight,

                  s.score_prime,
                  s.source,

                  {cast_review_count},
                  s.distance_km,

                  CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                       THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) END,
                  s.content_hash,
                  now()
                FROM {STG} s
                LEFT JOIN {TABLE} p ON p.source_id = s.source_id
                WHERE p.source_id IS NULL;
            """
            )
            inserted = cur.rowcount

            # 3) update changés
            # set_clause sans les colonnes à cast int
            set_clause = ",\n                  ".join(
                [
                    f"{c} = s.{c}"
                    for c in BASE_COLS
                    if c not in ("source_id",) and c not in CAST_INT_COLS
                ]
            )

            cur.execute(
                f"""
                UPDATE {TABLE} p
                SET
                  {set_clause},

                  region_insee = {cast_region_insee},
                  dept_insee   = {cast_dept_insee},
                  city_insee   = {cast_city_insee},
                  max_people   = {cast_max_people},
                  review_count = {cast_review_count},

                  geom = CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                              THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) END,
                  content_hash = s.content_hash,
                  updated_at = now()
                FROM {STG} s
                WHERE p.source_id = s.source_id
                  AND p.content_hash <> s.content_hash;
            """
            )
            updated = cur.rowcount

            # 4) SCD2 close
            cur.execute(
                f"""
                UPDATE {HIST_TABLE} h
                SET valid_to = now(), is_active = FALSE
                FROM {STG} s
                WHERE h.source_id = s.source_id
                  AND h.is_active = TRUE
                  AND h.content_hash <> s.content_hash;
            """
            )

            # 5) SCD2 insert nouveaux / changés (on écrit les valeurs déjà castées)
            cur.execute(
                f"""
                INSERT INTO {HIST_TABLE} (
                  source_id, valid_from, valid_to, is_active,
                  poi_types, name, label_en, snippet, snippet_en, lat, lon, country,
                  region_insee, region, dept_insee, departement, city_insee, city, postal_code,
                  address_locality, address,
                  theme_fr, theme_en, architectural_style_fr, architectural_style_en,
                  main_media_url, media_resource_url, url,
                  rating, review_value_label_fr, review_value_label_en, sub_category,
                  opens_time, closes_time, hours_valid_from, hours_valid_through,
                  max_people,
                  difficulty_level_fr, locomotion_mode_fr, tour_type_fr,
                  tour_distance_m, duration_min, practice_duration_min, duration_days, practice_duration_days,
                  positive_elevation_gain_m, negative_elevation_loss_m,
                  start_date, end_date, last_update_datatourisme,
                  price, min_price, max_price,
                  type_principal,
                  is_resto_type, is_resto_label, is_resto,
                  is_label_incontournable, is_label_famille, is_label_handicap, is_label_hebergement,
                  is_label_gastronomie, is_label_artisanat, is_label_randonnee, is_label_green, is_etoile,
                  price_level, main_category_compressed, is_prime_plus,
                  main_cat_weight, main_cat_candidates, format_label, format_weight, tempo_label, tempo_weight,
                  score_prime, source,
                  review_count, distance_km,
                  geom, content_hash
                )
                SELECT
                  s.source_id, now(), NULL, TRUE,
                  s.poi_types, s.name, s.label_en, s.snippet, s.snippet_en, s.lat, s.lon, s.country,
                  {cast_region_insee}, s.region, {cast_dept_insee}, s.departement, {cast_city_insee}, s.city, s.postal_code,
                  s.address_locality, s.address,
                  s.theme_fr, s.theme_en, s.architectural_style_fr, s.architectural_style_en,
                  s.main_media_url, s.media_resource_url, s.url,
                  s.rating, s.review_value_label_fr, s.review_value_label_en, s.sub_category,
                  s.opens_time, s.closes_time, s.hours_valid_from, s.hours_valid_through,
                  {cast_max_people},
                  s.difficulty_level_fr, s.locomotion_mode_fr, s.tour_type_fr,
                  s.tour_distance_m, s.duration_min, s.practice_duration_min, s.duration_days, s.practice_duration_days,
                  s.positive_elevation_gain_m, s.negative_elevation_loss_m,
                  s.start_date, s.end_date, s.last_update_datatourisme,
                  s.price, s.min_price, s.max_price,
                  s.type_principal,
                  s.is_resto_type, s.is_resto_label, s.is_resto,
                  s.is_label_incontournable, s.is_label_famille, s.is_label_handicap, s.is_label_hebergement,
                  s.is_label_gastronomie, s.is_label_artisanat, s.is_label_randonnee, s.is_label_green, s.is_etoile,
                  s.price_level, s.main_category_compressed, s.is_prime_plus,
                  s.main_cat_weight, s.main_cat_candidates, s.format_label, s.format_weight, s.tempo_label, s.tempo_weight,
                  s.score_prime, s.source,
                  {cast_review_count}, s.distance_km,
                  CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                       THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) END,
                  s.content_hash
                FROM {STG} s
                LEFT JOIN {HIST_TABLE} h
                  ON h.source_id = s.source_id AND h.is_active = TRUE
                WHERE h.source_id IS NULL OR h.content_hash <> s.content_hash;
            """
            )
            hist_added = cur.rowcount

            cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
            total = cur.fetchone()[0]

            print(
                f"[Prime Classique] inserted={inserted} updated={updated} history_rows_added={hist_added} total_now={total}"
            )


if __name__ == "__main__":
    main()
