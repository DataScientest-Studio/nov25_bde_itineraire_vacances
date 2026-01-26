import hashlib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

PARQUET_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances3\df_airbnb_Paris.parquet"

DSN = "host=localhost port=5432 dbname=itineraire_vacances_prime user=thuvo password=admin"

# Colonnes business pour le hash (tout sauf geom/updated_at)
HASH_COLS = [
    "source_id", "source", "type", "url", "lat", "lon", "name", "address", "postal_code", "city",
    "region", "country", "snippet", "rating", "review_count", "price_level", "max_people",
    "distance_km", "price"
]

def stable_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def compute_hash_row(row) -> str:
    s = "|".join(stable_str(row[c]) for c in HASH_COLS)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main():
    df = pd.read_parquet(PARQUET_PATH)

    # --- nettoyages minimums ---
    # ton "address" est float64 → on force en texte
    if "address" in df.columns:
        df["address"] = df["address"].astype("string").fillna("")

    # sécurité types
    df["source_id"] = df["source_id"].astype("int64")

    # hash
    df["content_hash"] = df.apply(compute_hash_row, axis=1)

    # colonnes à pousser en staging
    cols = HASH_COLS + ["content_hash"]

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            # 1) staging temp
            cur.execute("""
                DROP TABLE IF EXISTS _stg_airbnb;
                CREATE TEMP TABLE _stg_airbnb (
                  source_id     BIGINT,
                  source        TEXT,
                  type          TEXT,
                  url           TEXT,
                  lat           DOUBLE PRECISION,
                  lon           DOUBLE PRECISION,
                  name          TEXT,
                  address       TEXT,
                  postal_code   TEXT,
                  city          TEXT,
                  region        TEXT,
                  country       TEXT,
                  snippet       TEXT,
                  rating        DOUBLE PRECISION,
                  review_count  BIGINT,
                  price_level   DOUBLE PRECISION,
                  max_people    BIGINT,
                  distance_km   DOUBLE PRECISION,
                  price         DOUBLE PRECISION,
                  content_hash  TEXT
                ) ON COMMIT DROP;
            """)

            execute_values(
                cur,
                """
                INSERT INTO _stg_airbnb (
                  source_id, source, type, url, lat, lon, name, address, postal_code, city,
                  region, country, snippet, rating, review_count, price_level, max_people,
                  distance_km, price, content_hash
                ) VALUES %s
                """,
                df[cols].itertuples(index=False, name=None),
                page_size=5000
            )

            # 2) INSERT nouveaux
            cur.execute("""
                INSERT INTO silver.airbnb_paris (
                  source_id, source, type, url, lat, lon, geom, name, address, postal_code, city,
                  region, country, snippet, rating, review_count, price_level, max_people,
                  distance_km, price, content_hash, updated_at
                )
                SELECT
                  s.source_id, s.source, s.type, s.url, s.lat, s.lon,
                  CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                       THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326)::geometry(Point,4326)
                       ELSE NULL END,
                  s.name, s.address, s.postal_code, s.city,
                  s.region, s.country, s.snippet, s.rating, s.review_count, s.price_level, s.max_people,
                  s.distance_km, s.price, s.content_hash, now()
                FROM _stg_airbnb s
                LEFT JOIN silver.airbnb_paris p ON p.source_id = s.source_id
                WHERE p.source_id IS NULL;
            """)
            inserted = cur.rowcount

            # 3) UPDATE changés
            cur.execute("""
                UPDATE silver.airbnb_paris p
                SET
                  source = s.source,
                  type = s.type,
                  url = s.url,
                  lat = s.lat,
                  lon = s.lon,
                  geom = CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                              THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326)::geometry(Point,4326)
                              ELSE NULL END,
                  name = s.name,
                  address = s.address,
                  postal_code = s.postal_code,
                  city = s.city,
                  region = s.region,
                  country = s.country,
                  snippet = s.snippet,
                  rating = s.rating,
                  review_count = s.review_count,
                  price_level = s.price_level,
                  max_people = s.max_people,
                  distance_km = s.distance_km,
                  price = s.price,
                  content_hash = s.content_hash,
                  updated_at = now()
                FROM _stg_airbnb s
                WHERE p.source_id = s.source_id
                  AND p.content_hash <> s.content_hash;
            """)
            updated = cur.rowcount

            # 4) SCD2 : fermer l'actif si changement
            cur.execute("""
                UPDATE silver.airbnb_paris_history h
                SET valid_to = now(), is_active = FALSE
                FROM _stg_airbnb s
                WHERE h.source_id = s.source_id
                  AND h.is_active = TRUE
                  AND h.content_hash <> s.content_hash;
            """)

            # 5) SCD2 : insérer version active si nouveau ou changé
            cur.execute("""
                INSERT INTO silver.airbnb_paris_history (
                  source_id, valid_from, valid_to, is_active,
                  source, type, url, lat, lon, geom, name, address, postal_code, city, region,
                  country, snippet, rating, review_count, price_level, max_people, distance_km,
                  price, content_hash
                )
                SELECT
                  s.source_id, now(), NULL, TRUE,
                  s.source, s.type, s.url, s.lat, s.lon,
                  CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                       THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326)::geometry(Point,4326)
                       ELSE NULL END,
                  s.name, s.address, s.postal_code, s.city, s.region,
                  s.country, s.snippet, s.rating, s.review_count, s.price_level, s.max_people,
                  s.distance_km, s.price, s.content_hash
                FROM _stg_airbnb s
                LEFT JOIN silver.airbnb_paris_history h
                  ON h.source_id = s.source_id AND h.is_active = TRUE
                WHERE h.source_id IS NULL OR h.content_hash <> s.content_hash;
            """)
            hist_added = cur.rowcount

            cur.execute("SELECT COUNT(*) FROM silver.airbnb_paris;")
            total = cur.fetchone()[0]

            print(f"[Airbnb] inserted={inserted} updated={updated} history_rows_added={hist_added} total_now={total}")

if __name__ == "__main__":
    main()
