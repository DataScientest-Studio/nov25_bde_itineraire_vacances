-- 043_silver_tripadvisor_constraints.sql
-- Constraints for SILVER TripAdvisor tables:
--   - silver.tripadvisor_france
--   - silver.tripadvisor_france_history
-- Idempotent / safe to rerun

-- =========================================================
-- silver.tripadvisor_france : CHECK constraints (soft)
-- =========================================================
DO $$
BEGIN
  -- source_id non blank
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_source_id_nonblank'
  ) THEN
    ALTER TABLE silver.tripadvisor_france
      ADD CONSTRAINT ck_tripadvisor_source_id_nonblank
      CHECK (source_id IS NOT NULL AND btrim(source_id) <> '');
  END IF;

  -- lat/lon ranges
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_lat_range'
  ) THEN
    ALTER TABLE silver.tripadvisor_france
      ADD CONSTRAINT ck_tripadvisor_lat_range
      CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90));
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_lon_range'
  ) THEN
    ALTER TABLE silver.tripadvisor_france
      ADD CONSTRAINT ck_tripadvisor_lon_range
      CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180));
  END IF;

  -- reco_score non-negative + guardrail
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_reco_score_nonneg'
  ) THEN
    ALTER TABLE silver.tripadvisor_france
      ADD CONSTRAINT ck_tripadvisor_reco_score_nonneg
      CHECK (reco_score IS NULL OR reco_score >= 0);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_reco_score_reasonable'
  ) THEN
    ALTER TABLE silver.tripadvisor_france
      ADD CONSTRAINT ck_tripadvisor_reco_score_reasonable
      CHECK (reco_score IS NULL OR reco_score < 100);
  END IF;

  -- distance_km guardrails (placeholder OK)
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_distance_nonneg'
  ) THEN
    ALTER TABLE silver.tripadvisor_france
      ADD CONSTRAINT ck_tripadvisor_distance_nonneg
      CHECK (distance_km IS NULL OR distance_km >= 0);
  END IF;

END$$;

-- =========================================================
-- silver.tripadvisor_france_history : CHECK constraints (soft)
-- =========================================================
DO $$
BEGIN
  -- SCD2 consistency: is_active <-> valid_to
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_hist_active_valid_to'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT ck_tripadvisor_hist_active_valid_to
      CHECK (
        (is_active = TRUE AND valid_to IS NULL)
        OR (is_active = FALSE AND valid_to IS NOT NULL)
      );
  END IF;

  -- lat/lon ranges
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_hist_lat_range'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT ck_tripadvisor_hist_lat_range
      CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90));
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_hist_lon_range'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT ck_tripadvisor_hist_lon_range
      CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180));
  END IF;


  -- reco_score guardrails
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_hist_reco_score_nonneg'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT ck_tripadvisor_hist_reco_score_nonneg
      CHECK (reco_score IS NULL OR reco_score >= 0);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_hist_reco_score_reasonable'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT ck_tripadvisor_hist_reco_score_reasonable
      CHECK (reco_score IS NULL OR reco_score < 100);
  END IF;


  -- distance_km guardrails
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_tripadvisor_hist_distance_nonneg'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT ck_tripadvisor_hist_distance_nonneg
      CHECK (distance_km IS NULL OR distance_km >= 0);
  END IF;

END$$;

-- =========================================================
-- FK (history -> current)
-- =========================================================
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'fk_tripadvisor_history_source_id'
  ) THEN
    ALTER TABLE silver.tripadvisor_france_history
      ADD CONSTRAINT fk_tripadvisor_history_source_id
      FOREIGN KEY (source_id)
      REFERENCES silver.tripadvisor_france(source_id)
      DEFERRABLE INITIALLY DEFERRED;
  END IF;
END$$;
