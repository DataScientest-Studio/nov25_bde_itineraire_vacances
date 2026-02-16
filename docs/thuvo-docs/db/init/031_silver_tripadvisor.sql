-- 031_silver_tripadvisor.sql
-- SILVER layer for TripAdvisor (France)
-- Tables:
--   - silver.tripadvisor_france (current snapshot, 1 row/source_id)
--   - silver.tripadvisor_france_history (SCD2 history)

CREATE SCHEMA IF NOT EXISTS silver;
CREATE EXTENSION IF NOT EXISTS postgis;

-- =========================================================
-- CURRENT TABLE: silver.tripadvisor_france
-- =========================================================
CREATE TABLE IF NOT EXISTS silver.tripadvisor_france (
  -- Business key
  source_id TEXT PRIMARY KEY,

  -- Raw lineage
  raw_loaded_at TIMESTAMPTZ,
  raw_source_file TEXT,

  -- Identifiants / source
  source TEXT,

  -- Typage / URL
  type TEXT,
  url TEXT,

  -- Géoloc
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  geom geometry(Point, 4326),

  -- Identité / adresse
  name TEXT,
  address TEXT,
  postal_code TEXT,
  city TEXT,
  region TEXT,
  country TEXT,

  -- Contenu
  snippet TEXT,

  -- Signaux
  rating DOUBLE PRECISION,
  review_count BIGINT,              -- normalisé (depuis float64 raw)
  reco_score DOUBLE PRECISION,

  -- Prix
  price_level TEXT,
  price_range TEXT,
  price DOUBLE PRECISION,

  -- Cuisines
  cuisine_continent TEXT,
  cuisine_type TEXT,

  -- Flags
  vegetarian_friendly BOOLEAN,
  vegan_options BOOLEAN,
  gluten_free BOOLEAN,
  is_halal BOOLEAN,
  is_kosher BOOLEAN,
  awards_binary BOOLEAN,

  -- Placeholders (si tu les alimentes plus tard)
  max_people DOUBLE PRECISION,
  distance_km DOUBLE PRECISION,

  -- Ops / SCD2
  content_hash TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================================================
-- HISTORY TABLE: silver.tripadvisor_france_history (SCD2)
-- =========================================================
CREATE TABLE IF NOT EXISTS silver.tripadvisor_france_history (
  history_id BIGSERIAL PRIMARY KEY,
  source_id TEXT NOT NULL,

  valid_from TIMESTAMPTZ NOT NULL,
  valid_to   TIMESTAMPTZ NULL,
  is_active  BOOLEAN NOT NULL DEFAULT TRUE,

  -- Raw lineage
  raw_loaded_at TIMESTAMPTZ,
  raw_source_file TEXT,

  source TEXT,
  type TEXT,
  url TEXT,

  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  geom geometry(Point, 4326),

  name TEXT,
  address TEXT,
  postal_code TEXT,
  city TEXT,
  region TEXT,
  country TEXT,

  snippet TEXT,

  rating DOUBLE PRECISION,
  review_count BIGINT,
  reco_score DOUBLE PRECISION,

  price_level TEXT,
  price_range TEXT,
  price DOUBLE PRECISION,

  cuisine_continent TEXT,
  cuisine_type TEXT,

  vegetarian_friendly BOOLEAN,
  vegan_options BOOLEAN,
  gluten_free BOOLEAN,
  is_halal BOOLEAN,
  is_kosher BOOLEAN,
  awards_binary BOOLEAN,

  max_people DOUBLE PRECISION,
  distance_km DOUBLE PRECISION,

  content_hash TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================================================
-- Minimal history helpers (indexes/constraints often split to 042/043)
-- =========================================================

-- One active row per source_id (SCD2 safety)
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_tripadvisor_hist_one_active
  ON silver.tripadvisor_france_history (source_id)
  WHERE is_active = TRUE;

-- Fast lookup by source_id + active
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_hist_source_active
  ON silver.tripadvisor_france_history (source_id, is_active);
