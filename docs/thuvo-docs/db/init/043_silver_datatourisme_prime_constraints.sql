-- 043_silver_datatourisme_prime_constraints.sql
-- Constraints for SILVER DataTourisme Prime:
--   - silver.datatourisme_prime
--   - silver.datatourisme_prime_history (optional SCD2)

-- =========================================================
-- silver.datatourisme_prime : CHECK constraints (soft)
-- =========================================================
DO $$
BEGIN
  IF to_regclass('silver.datatourisme_prime') IS NULL THEN
    RAISE NOTICE 'Skip constraints: table silver.datatourisme_prime does not exist';
    RETURN;
  END IF;

  -- source_id non blank
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_source_id_nonblank') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_source_id_nonblank
      CHECK (source_id IS NOT NULL AND btrim(source_id) <> '');
  END IF;

  -- rating in [0,5]
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_rating_range') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_rating_range
      CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5));
  END IF;

  -- lat/lon ranges
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_lat_range') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_lat_range
      CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90));
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_lon_range') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_lon_range
      CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180));
  END IF;

  -- non-negative guards
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_distance_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_distance_nonneg
      CHECK (distance_km IS NULL OR distance_km >= 0);
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_score_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_score_nonneg
      CHECK (score_prime IS NULL OR score_prime >= 0);
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_main_cat_weight_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_main_cat_weight_nonneg
      CHECK (main_cat_weight IS NULL OR main_cat_weight >= 0);
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_format_weight_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_format_weight_nonneg
      CHECK (format_weight IS NULL OR format_weight >= 0);
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_tempo_weight_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_tempo_weight_nonneg
      CHECK (tempo_weight IS NULL OR tempo_weight >= 0);
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_review_count_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime
      ADD CONSTRAINT ck_dt_prime_review_count_nonneg
      CHECK (review_count IS NULL OR review_count >= 0);
  END IF;

END$$;


-- =========================================================
-- silver.datatourisme_prime_history : CHECK constraints (soft)
-- (optional table, do nothing if missing)
-- =========================================================
DO $$
BEGIN
  IF to_regclass('silver.datatourisme_prime_history') IS NULL THEN
    RAISE NOTICE 'Skip history constraints: table silver.datatourisme_prime_history does not exist';
    RETURN;
  END IF;

  -- is_active implies valid_to is NULL (SCD2)
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_hist_active_valid_to') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT ck_dt_prime_hist_active_valid_to
      CHECK (
        (is_active = TRUE  AND valid_to IS NULL)
        OR
        (is_active = FALSE AND valid_to IS NOT NULL)
      );
  END IF;

  -- rating in [0,5]
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_hist_rating_range') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT ck_dt_prime_hist_rating_range
      CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5));
  END IF;

  -- lat/lon ranges
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_hist_lat_range') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT ck_dt_prime_hist_lat_range
      CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90));
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_hist_lon_range') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT ck_dt_prime_hist_lon_range
      CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180));
  END IF;

  -- non-negative guards
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_hist_distance_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT ck_dt_prime_hist_distance_nonneg
      CHECK (distance_km IS NULL OR distance_km >= 0);
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_dt_prime_hist_review_count_nonneg') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT ck_dt_prime_hist_review_count_nonneg
      CHECK (review_count IS NULL OR review_count >= 0);
  END IF;

  -- FK (history -> current)
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_dt_prime_history_source_id') THEN
    ALTER TABLE silver.datatourisme_prime_history
      ADD CONSTRAINT fk_dt_prime_history_source_id
      FOREIGN KEY (source_id)
      REFERENCES silver.datatourisme_prime(source_id)
      DEFERRABLE INITIALLY DEFERRED;
  END IF;

END$$;
