-- ============================================================
-- init.sql (version stable)
-- La DB + l'utilisateur sont créés par docker-compose via :
-- POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
--
-- Ici on fait seulement :
-- - PostGIS
-- - schéma silver
-- - tables Airbnb + history
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE IF NOT EXISTS silver.airbnb_paris (
  source_id     BIGINT PRIMARY KEY,
  source        TEXT,
  type          TEXT,
  url           TEXT,
  lat           DOUBLE PRECISION,
  lon           DOUBLE PRECISION,
  geom          geometry(Point, 4326),
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
  content_hash  TEXT NOT NULL,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_airbnb_geom
ON silver.airbnb_paris USING GIST(geom);

CREATE TABLE IF NOT EXISTS silver.airbnb_paris_history (
  history_id    BIGSERIAL PRIMARY KEY,
  source_id     BIGINT NOT NULL,
  valid_from    TIMESTAMPTZ NOT NULL,
  valid_to      TIMESTAMPTZ,
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,

  source        TEXT,
  type          TEXT,
  url           TEXT,
  lat           DOUBLE PRECISION,
  lon           DOUBLE PRECISION,
  geom          geometry(Point, 4326),
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

  content_hash  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_airbnb_hist_active
ON silver.airbnb_paris_history(source_id)
WHERE is_active;

CREATE INDEX IF NOT EXISTS idx_airbnb_hist_geom
ON silver.airbnb_paris_history USING GIST(geom);
