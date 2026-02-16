-- 042_silver_tripadvisor_indexes.sql
-- Indexes for SILVER TripAdvisor tables:
--   - silver.tripadvisor_france
--   - silver.tripadvisor_france_history
-- Idempotent / safe to rerun

-- =========================================================
-- CURRENT TABLE: silver.tripadvisor_france
-- =========================================================

-- Spatial index (PostGIS)
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_geom
  ON silver.tripadvisor_france USING GIST (geom);

-- Common filters
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_city
  ON silver.tripadvisor_france (city);

CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_region
  ON silver.tripadvisor_france (region);

CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_price_level
  ON silver.tripadvisor_france (price_level);

-- Sorting / ranking
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_reco_score
  ON silver.tripadvisor_france (reco_score);

-- Optional (si tu filtres souvent dessus)
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_type
  ON silver.tripadvisor_france (type);

CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_cuisine_continent
  ON silver.tripadvisor_france (cuisine_continent);

CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_cuisine_type
  ON silver.tripadvisor_france (cuisine_type);

-- =========================================================
-- HISTORY TABLE: silver.tripadvisor_france_history (SCD2)
-- =========================================================

-- One active row per source_id (partial unique index)
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_tripadvisor_hist_one_active
  ON silver.tripadvisor_france_history (source_id)
  WHERE is_active = TRUE;

-- Fast lookup by (source_id, is_active)
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_hist_source_active
  ON silver.tripadvisor_france_history (source_id, is_active);

-- Time-based audits
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_hist_valid_from
  ON silver.tripadvisor_france_history (valid_from);

CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_hist_valid_to
  ON silver.tripadvisor_france_history (valid_to);

-- Spatial history
CREATE INDEX IF NOT EXISTS idx_silver_tripadvisor_hist_geom
  ON silver.tripadvisor_france_history USING GIST (geom);
