import hashlib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

PARQUET_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances3\df_tripadvisor_France.parquet"

DSN = "host=localhost port=5432 dbname=itineraire_vacances_prime user=thuvo password=admin"

HASH_COLS = [
    "source_id", "source", "type", "url", "lat", "lon", "name", "address",
    "postal_code", "city", "region", "country", "snippet", "rating",
    "review_count", "price_level", "reco_score", "cuisine_continent",
    "cuisine_type", "vegetarian_friendly", "vegan_options", "gluten_free",
    "is_halal", "is_kosher", "max_people", "distance_km"
]

def stable_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def compute_hash_row(row):
    s = "|".join(stable_str(row[c]) for c in HASH_COLS)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main():
    df = pd.read_parquet(PARQUET_PATH)

    # --- max_people : nettoyage robuste ---
    if "max_people" in df.columns:
        df["max_people"] = pd.to_numeric(df["max_people"], errors="coerce")

        # on considère qu'un resto n'a jamais > 10 000 places (valeur garde-fou)
        df.loc[(df["max_people"] < 0) | (df["max_people"] > 10000), "max_people"] = pd.NA

        # Convertir en entier nullable pandas
        df["max_people"] = df["max_people"].astype("Int64")

    # types de base
    df["source_id"] = df["source_id"].astype(str)
    df["address"] = df["address"].astype("string").fillna("")

    # bool safety
    for c in ["vegetarian_friendly", "vegan_options", "gluten_free", "is_halal", "is_kosher"]:
        if c in df.columns:
            df[c] = df[c].fillna(False)

    df["content_hash"] = df.apply(compute_hash_row, axis=1)

    cols = HASH_COLS + ["content_hash"]

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:

            # staging temp
            cur.execute("""
                DROP TABLE IF EXISTS _stg_tripadvisor;
                CREATE TEMP TABLE _stg_tripadvisor (
                  source_id TEXT,
                  source TEXT,
                  type TEXT,
                  url TEXT,
                  lat DOUBLE PRECISION,
                  lon DOUBLE PRECISION,
                  name TEXT,
                  address TEXT,
                  postal_code TEXT,
                  city TEXT,
                  region TEXT,
                  country TEXT,
                  snippet TEXT,
                  rating DOUBLE PRECISION,
                  review_count DOUBLE PRECISION,
                  price_level DOUBLE PRECISION,
                  reco_score DOUBLE PRECISION,
                  cuisine_continent TEXT,
                  cuisine_type TEXT,
                  vegetarian_friendly BOOLEAN,
                  vegan_options BOOLEAN,
                  gluten_free BOOLEAN,
                  is_halal BOOLEAN,
                  is_kosher BOOLEAN,
                  max_people BIGINT,
                  distance_km DOUBLE PRECISION,
                  content_hash TEXT
                ) ON COMMIT DROP;
            """)

            bad = df[df["max_people"].notna() & (df["max_people"] > 10000)]
            print("max_people > 10000:", len(bad))
            if len(bad) > 0:
                print(bad[["source_id", "max_people"]].head(10))
            
            # Convertit tous les NA/NaT pandas en None (NULL SQL)
            df_to_load = df[cols].copy()
            df_to_load = df_to_load.astype(object).where(pd.notna(df_to_load), None)

            # IMPORTANT: on fabrique des tuples Python "propres"
            rows = [tuple(r) for r in df_to_load.to_numpy()]

            insert_sql = """
                INSERT INTO _stg_tripadvisor (
                  source_id, source, type, url, lat, lon, name, address,
                  postal_code, city, region, country, snippet, rating,
                  review_count, price_level, reco_score, cuisine_continent,
                  cuisine_type, vegetarian_friendly, vegan_options,
                  gluten_free, is_halal, is_kosher, max_people, distance_km,
                  content_hash
                ) VALUES %s
            """

            execute_values(
                cur,
                insert_sql,
                rows,
                page_size=5000)

            # insert nouveaux
            cur.execute("""
                INSERT INTO silver.tripadvisor_france (
                  source_id, source, type, url, lat, lon, geom, name, address,
                  postal_code, city, region, country, snippet, rating,
                  review_count, price_level, reco_score, cuisine_continent,
                  cuisine_type, vegetarian_friendly, vegan_options,
                  gluten_free, is_halal, is_kosher, max_people, distance_km,
                  content_hash, updated_at
                )
                SELECT
                  s.source_id, s.source, s.type, s.url, s.lat, s.lon,
                  CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                       THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) END,
                  s.name, s.address, s.postal_code, s.city, s.region, s.country,
                  s.snippet, s.rating, s.review_count, s.price_level,
                  s.reco_score, s.cuisine_continent, s.cuisine_type,
                  s.vegetarian_friendly, s.vegan_options, s.gluten_free,
                  s.is_halal, s.is_kosher, s.max_people, s.distance_km,
                  s.content_hash, now()
                FROM _stg_tripadvisor s
                LEFT JOIN silver.tripadvisor_france t
                  ON t.source_id = s.source_id
                WHERE t.source_id IS NULL;
            """)
            inserted = cur.rowcount

            # update changés
            cur.execute("""
                UPDATE silver.tripadvisor_france t
                SET
                  source = s.source,
                  type = s.type,
                  url = s.url,
                  lat = s.lat,
                  lon = s.lon,
                  geom = CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                              THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) END,
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
                  reco_score = s.reco_score,
                  cuisine_continent = s.cuisine_continent,
                  cuisine_type = s.cuisine_type,
                  vegetarian_friendly = s.vegetarian_friendly,
                  vegan_options = s.vegan_options,
                  gluten_free = s.gluten_free,
                  is_halal = s.is_halal,
                  is_kosher = s.is_kosher,
                  max_people = s.max_people,
                  distance_km = s.distance_km,
                  content_hash = s.content_hash,
                  updated_at = now()
                FROM _stg_tripadvisor s
                WHERE t.source_id = s.source_id
                  AND t.content_hash <> s.content_hash;
            """)
            updated = cur.rowcount

            # SCD2 close
            cur.execute("""
                UPDATE silver.tripadvisor_france_history h
                SET valid_to = now(), is_active = FALSE
                FROM _stg_tripadvisor s
                WHERE h.source_id = s.source_id
                  AND h.is_active = TRUE
                  AND h.content_hash <> s.content_hash;
            """)

            # SCD2 insert
            cur.execute("""
                INSERT INTO silver.tripadvisor_france_history (
                  source_id, valid_from, valid_to, is_active,
                  source, type, url, lat, lon, geom, name, address,
                  postal_code, city, region, country, snippet, rating,
                  review_count, price_level, reco_score, cuisine_continent,
                  cuisine_type, vegetarian_friendly, vegan_options,
                  gluten_free, is_halal, is_kosher, max_people, distance_km,
                  content_hash
                )
                SELECT
                  s.source_id, now(), NULL, TRUE,
                  s.source, s.type, s.url, s.lat, s.lon,
                  CASE WHEN s.lon IS NOT NULL AND s.lat IS NOT NULL
                       THEN ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) END,
                  s.name, s.address, s.postal_code, s.city, s.region, s.country,
                  s.snippet, s.rating, s.review_count, s.price_level,
                  s.reco_score, s.cuisine_continent, s.cuisine_type,
                  s.vegetarian_friendly, s.vegan_options, s.gluten_free,
                  s.is_halal, s.is_kosher, s.max_people, s.distance_km,
                  s.content_hash
                FROM _stg_tripadvisor s
                LEFT JOIN silver.tripadvisor_france_history h
                  ON h.source_id = s.source_id AND h.is_active = TRUE
                WHERE h.source_id IS NULL OR h.content_hash <> s.content_hash;
            """)
            hist_added = cur.rowcount

            cur.execute("SELECT COUNT(*) FROM silver.tripadvisor_france;")
            total = cur.fetchone()[0]

            print(f"[TripAdvisor] inserted={inserted} updated={updated} history_rows_added={hist_added} total_now={total}")

if __name__ == "__main__":
    main()
