-- 021_raw_tripadvisor.sql
-- RAW layer for TripAdvisor (France)
-- Mirror of parquet columns (no business logic)

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.tripadvisor_france (

  -- Identifiants / source
  source_id TEXT,
  source TEXT,

  -- Typage / URL
  type TEXT,
  url TEXT,

  -- Géoloc
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,

  -- Identité / adresse
  name TEXT,
  address TEXT,
  postal_code TEXT,
  city TEXT,
  region TEXT,
  country TEXT,

  -- Contenu & signaux
  snippet TEXT,
  rating DOUBLE PRECISION,
  review_count DOUBLE PRECISION,   -- parquet=float64 (on normalisera en SILVER si besoin)

  -- Prix
  price_level TEXT,
  price_range TEXT,
  price DOUBLE PRECISION,

  -- Reco & cuisines
  reco_score DOUBLE PRECISION,
  cuisine_continent TEXT,
  cuisine_type TEXT,

  -- Flags
  vegetarian_friendly BOOLEAN,
  vegan_options BOOLEAN,
  gluten_free BOOLEAN,
  is_halal BOOLEAN,
  is_kosher BOOLEAN,
  awards_binary BOOLEAN,

  -- Placeholders (présents dans parquet mais vides)
  max_people DOUBLE PRECISION,
  distance_km DOUBLE PRECISION,

  -- Meta ingestion (RAW only)
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Meta ingestion (RAW only)
  raw_loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  raw_source_file TEXT
);

-- Index utiles pour filtres fréquents
CREATE INDEX IF NOT EXISTS idx_raw_tripadvisor_source_id
  ON raw.tripadvisor_france (source_id);

CREATE INDEX IF NOT EXISTS idx_raw_tripadvisor_city
  ON raw.tripadvisor_france (city);

CREATE INDEX IF NOT EXISTS idx_raw_tripadvisor_type
  ON raw.tripadvisor_france (type);
