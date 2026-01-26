CREATE TABLE IF NOT EXISTS silver.tripadvisor_france (
  source_id           TEXT PRIMARY KEY,
  source              TEXT,
  type                TEXT,
  url                 TEXT,
  lat                 DOUBLE PRECISION,
  lon                 DOUBLE PRECISION,
  geom                geometry(Point, 4326),
  name                TEXT,
  address             TEXT,
  postal_code         TEXT,
  city                TEXT,
  region              TEXT,
  country             TEXT,
  snippet             TEXT,
  rating              DOUBLE PRECISION,
  review_count        DOUBLE PRECISION,
  price_level         DOUBLE PRECISION,
  reco_score          DOUBLE PRECISION,
  cuisine_continent   TEXT,
  cuisine_type        TEXT,
  vegetarian_friendly BOOLEAN,
  vegan_options       BOOLEAN,
  gluten_free         BOOLEAN,
  is_halal            BOOLEAN,
  is_kosher           BOOLEAN,
  max_people          BIGINT,
  distance_km         DOUBLE PRECISION,
  content_hash        TEXT NOT NULL,
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tripadvisor_geom
ON silver.tripadvisor_france USING GIST(geom);

CREATE TABLE IF NOT EXISTS silver.tripadvisor_france_history (
  history_id          BIGSERIAL PRIMARY KEY,
  source_id           TEXT NOT NULL,
  valid_from          TIMESTAMPTZ NOT NULL,
  valid_to            TIMESTAMPTZ,
  is_active           BOOLEAN NOT NULL DEFAULT TRUE,

  source              TEXT,
  type                TEXT,
  url                 TEXT,
  lat                 DOUBLE PRECISION,
  lon                 DOUBLE PRECISION,
  geom                geometry(Point, 4326),
  name                TEXT,
  address             TEXT,
  postal_code         TEXT,
  city                TEXT,
  region              TEXT,
  country             TEXT,
  snippet             TEXT,
  rating              DOUBLE PRECISION,
  review_count        DOUBLE PRECISION,
  price_level         DOUBLE PRECISION,
  reco_score          DOUBLE PRECISION,
  cuisine_continent   TEXT,
  cuisine_type        TEXT,
  vegetarian_friendly BOOLEAN,
  vegan_options       BOOLEAN,
  gluten_free         BOOLEAN,
  is_halal            BOOLEAN,
  is_kosher           BOOLEAN,
  max_people          BIGINT,
  distance_km         DOUBLE PRECISION,
  content_hash        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tripadvisor_hist_active
ON silver.tripadvisor_france_history(source_id)
WHERE is_active;

CREATE INDEX IF NOT EXISTS idx_tripadvisor_hist_geom
ON silver.tripadvisor_france_history USING GIST(geom);
