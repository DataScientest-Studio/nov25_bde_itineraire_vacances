-- 041_silver_datatourisme_prime.sql
-- SILVER (current) for DataTourisme Prime
-- Source: raw.datatourisme_prime
-- Fixes:
-- - Ne référence JAMAIS les colonnes de la table cible dans le SELECT (ex: updated_at)
-- - Hash stable (idempotent) basé sur champs source + last_update_datatourisme
-- - Casts sûrs pour bool/text

CREATE SCHEMA IF NOT EXISTS silver;

-- =========================================================
-- CURRENT TABLE: silver.datatourisme_prime
-- =========================================================
CREATE TABLE IF NOT EXISTS silver.datatourisme_prime (
  source_id TEXT PRIMARY KEY,

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
  media_resource_url TEXT,
  url TEXT,

  rating DOUBLE PRECISION,
  review_value_label_fr TEXT,
  review_value_label_en TEXT,
  sub_category TEXT,

  max_people INTEGER,
  difficulty_level_fr TEXT,
  locomotion_mode_fr TEXT,
  tour_type_fr TEXT,
  last_update_datatourisme TIMESTAMPTZ,

  price DOUBLE PRECISION,
  price_level TEXT,
  price_range TEXT,

  is_label_incontournable BOOLEAN NOT NULL,
  is_label_handicap BOOLEAN NOT NULL,
  is_label_hebergement BOOLEAN NOT NULL,
  is_label_gastronomy BOOLEAN NOT NULL,
  is_label_artisanat BOOLEAN NOT NULL,
  is_label_itinerary BOOLEAN NOT NULL,
  is_label_green BOOLEAN NOT NULL,

  main_category TEXT,
  main_cat_weight DOUBLE PRECISION,
  type_principal TEXT,
  format_weight DOUBLE PRECISION,
  tempo_weight DOUBLE PRECISION,
  score_prime DOUBLE PRECISION,
  hebergement_type TEXT,

  tour_distance_bucket TEXT,
  practice_duration_final_bucket TEXT,
  seasons_available TEXT,
  opening_hours_str TEXT,
  elevation_gain_loss TEXT,

  source TEXT,

  review_count INTEGER,
  distance_km DOUBLE PRECISION,

  geom geometry(Point, 4326),
  content_hash TEXT,
  updated_at TIMESTAMPTZ
);

-- =========================================================
-- LOAD CURRENT TABLE from raw.datatourisme_prime
-- =========================================================

TRUNCATE TABLE silver.datatourisme_prime;

INSERT INTO silver.datatourisme_prime (
  source_id,
  name, label_en, snippet, snippet_en,
  lat, lon,
  country, region_insee, region, dept_insee, departement, city_insee, city,
  postal_code, address_locality, address,
  theme_fr, theme_en, media_resource_url, url,
  rating, review_value_label_fr, review_value_label_en, sub_category,
  max_people, difficulty_level_fr, locomotion_mode_fr, tour_type_fr, last_update_datatourisme,
  price, price_level, price_range,
  is_label_incontournable, is_label_handicap, is_label_hebergement, is_label_gastronomy,
  is_label_artisanat, is_label_itinerary, is_label_green,
  main_category, main_cat_weight, type_principal, format_weight, tempo_weight,
  score_prime, hebergement_type,
  tour_distance_bucket, practice_duration_final_bucket, seasons_available, opening_hours_str, elevation_gain_loss,
  source,
  review_count, distance_km,
  geom, content_hash, updated_at
)
SELECT
  r.source_id,
  r.name, r.label_en, r.snippet, r.snippet_en,
  r.lat::double precision,
  r.lon::double precision,
  r.country, r.region_insee, r.region, r.dept_insee, r.departement, r.city_insee, r.city,
  r.postal_code, r.address_locality, r.address,
  r.theme_fr, r.theme_en, r.media_resource_url, r.url,
  r.rating::double precision,
  r.review_value_label_fr, r.review_value_label_en, r.sub_category,
  r.max_people, r.difficulty_level_fr, r.locomotion_mode_fr, r.tour_type_fr, r.last_update_datatourisme,
  r.price::double precision, r.price_level, r.price_range,

  COALESCE(NULLIF(lower(r.is_label_incontournable::text), '')::boolean, FALSE) AS is_label_incontournable,
  COALESCE(NULLIF(lower(r.is_label_handicap::text), '')::boolean, FALSE)       AS is_label_handicap,
  COALESCE(NULLIF(lower(r.is_label_hebergement::text), '')::boolean, FALSE)    AS is_label_hebergement,
  COALESCE(NULLIF(lower(r.is_label_gastronomy::text), '')::boolean, FALSE)     AS is_label_gastronomy,
  COALESCE(NULLIF(lower(r.is_label_artisanat::text), '')::boolean, FALSE)      AS is_label_artisanat,
  COALESCE(NULLIF(lower(r.is_label_itinerary::text), '')::boolean, FALSE)      AS is_label_itinerary,
  COALESCE(NULLIF(lower(r.is_label_green::text), '')::boolean, FALSE)          AS is_label_green,

  r.main_category,
  r.main_cat_weight::double precision,
  r.type_principal,
  r.format_weight::double precision,
  r.tempo_weight::double precision,
  r.score_prime::double precision,
  r.hebergement_type,

  r.tour_distance_bucket, r.practice_duration_final_bucket, r.seasons_available, r.opening_hours_str, r.elevation_gain_loss,
  r.source,
  r.review_count,
  r.distance_km,

  ST_SetSRID(ST_MakePoint(r.lon::double precision, r.lat::double precision), 4326) AS geom,

  -- Hash stable (idempotent) : NE PAS inclure updated_at (sinon change à chaque run)
  md5(
    coalesce(r.source_id::text,'') || '|' ||
    coalesce(r.name::text,'')      || '|' ||
    coalesce(r.lat::text,'')       || '|' ||
    coalesce(r.lon::text,'')       || '|' ||
    coalesce(r.last_update_datatourisme::text,'')
  ) AS content_hash,

  NOW() AS updated_at
FROM raw.datatourisme_prime AS r
WHERE r.source_id IS NOT NULL;
