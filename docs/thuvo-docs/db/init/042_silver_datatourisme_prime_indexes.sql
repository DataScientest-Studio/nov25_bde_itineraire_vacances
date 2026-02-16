-- 042_silver_datatourisme_prime_indexes.sql
-- Indexes for SILVER DataTourisme Prime tables:
--   - silver.datatourisme_prime
--   - silver.datatourisme_prime_history (optional SCD2)
--
-- Aligned with 041_silver_datatourisme_prime.sql

-- =========================================================
-- CURRENT TABLE: silver.datatourisme_prime
-- =========================================================

-- Spatial index (PostGIS)
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_geom
  ON silver.datatourisme_prime USING GIST (geom);

-- Common filters / group-by (UI)
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_main_category
  ON silver.datatourisme_prime (main_category);

CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_type_principal
  ON silver.datatourisme_prime (type_principal);

CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_city
  ON silver.datatourisme_prime (city);

-- Price filters
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_price_level
  ON silver.datatourisme_prime (price_level);

-- Prime sorting
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_score_prime
  ON silver.datatourisme_prime (score_prime);

-- Labels (boolean filters often used in UI)
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_label_incontournable
  ON silver.datatourisme_prime (is_label_incontournable);

CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_label_hebergement
  ON silver.datatourisme_prime (is_label_hebergement);

-- Itinerary-related filters (si utilisés dans la sidebar)
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_tour_type_fr
  ON silver.datatourisme_prime (tour_type_fr);

-- Optional: “fraîcheur” (audits / sync)
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_last_update_dt
  ON silver.datatourisme_prime (last_update_datatourisme);

-- Optional: accelerate SCD2 comparisons / dedup
CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_content_hash
  ON silver.datatourisme_prime (content_hash);


-- =========================================================
-- HISTORY TABLE: silver.datatourisme_prime_history (SCD2)
--   This table may not exist if SCD2 is not enabled.
--   All history indexes are created conditionally.
-- =========================================================
DO $$
BEGIN
  IF to_regclass('silver.datatourisme_prime_history') IS NOT NULL THEN

    -- Enforce one active row per source_id (partial unique index)
    EXECUTE $q$
      CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_dt_prime_hist_one_active
      ON silver.datatourisme_prime_history (source_id)
      WHERE is_active = TRUE
    $q$;

    -- Fast lookup / join by source_id + state
    EXECUTE $q$
      CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_hist_source_active
      ON silver.datatourisme_prime_history (source_id, is_active)
    $q$;

    -- Time-based queries / audits
    EXECUTE $q$
      CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_hist_valid_from
      ON silver.datatourisme_prime_history (valid_from)
    $q$;

    EXECUTE $q$
      CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_hist_valid_to
      ON silver.datatourisme_prime_history (valid_to)
    $q$;

    -- Spatial history (PostGIS)
    EXECUTE $q$
      CREATE INDEX IF NOT EXISTS idx_silver_dt_prime_hist_geom
      ON silver.datatourisme_prime_history USING GIST (geom)
    $q$;

  ELSE
    RAISE NOTICE 'Skip history indexes: table silver.datatourisme_prime_history does not exist';
  END IF;
END $$;
