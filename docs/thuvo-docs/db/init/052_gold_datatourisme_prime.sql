-- 052_gold_datatourisme_prime.sql
-- GOLD layer - DataTourisme Prime (serving contract + MV)
-- Compatible with current schema: silver.datatourisme_prime
-- (no hebergement_type, no is_active in silver -> set TRUE in GOLD)

CREATE SCHEMA IF NOT EXISTS gold;

-- =========================================================
-- 1) VIEW API-ready (always up-to-date)
-- =========================================================
CREATE OR REPLACE VIEW gold.v_datatourisme_prime_api AS
SELECT
  source_id,

  lat::double precision AS lat,
  lon::double precision AS lon,
  geom,

  city,
  region,

  rating::double precision AS rating,
  review_count,

  price_level,

  main_category,
  sub_category,
  type_principal,

  main_cat_weight::double precision AS main_cat_weight,
  format_weight::double precision   AS format_weight,
  tempo_weight::double precision    AS tempo_weight,

  score_prime::double precision     AS score_prime,

  distance_km::double precision     AS distance_km,

  TRUE::boolean AS is_active
FROM silver.datatourisme_prime;

-- =========================================================
-- 2) MV "serving" (fast filters / map)
-- =========================================================
DROP MATERIALIZED VIEW IF EXISTS gold.mv_datatourisme_prime_serving;

CREATE MATERIALIZED VIEW gold.mv_datatourisme_prime_serving AS
SELECT
  source_id,

  lat::double precision AS lat,
  lon::double precision AS lon,
  geom,

  city,
  region,

  rating::double precision AS rating,
  review_count,

  price_level,

  main_category,
  sub_category,
  type_principal,

  main_cat_weight::double precision AS main_cat_weight,
  format_weight::double precision   AS format_weight,
  tempo_weight::double precision    AS tempo_weight,

  score_prime::double precision     AS score_prime,

  distance_km::double precision     AS distance_km,

  TRUE::boolean AS is_active
FROM silver.datatourisme_prime
WHERE geom IS NOT NULL
  AND lat IS NOT NULL
  AND lon IS NOT NULL;

-- =========================================================
-- 3) Indexes (MV)
-- =========================================================
DROP INDEX IF EXISTS gold.ux_gold_mv_datatourisme_prime_serving_source_id;

CREATE UNIQUE INDEX ux_gold_mv_datatourisme_prime_serving_source_id
  ON gold.mv_datatourisme_prime_serving (source_id);

CREATE INDEX IF NOT EXISTS idx_gold_mv_datatourisme_prime_serving_geom
  ON gold.mv_datatourisme_prime_serving USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_gold_mv_datatourisme_prime_serving_filters
  ON gold.mv_datatourisme_prime_serving (
    city,
    region,
    main_category,
    sub_category,
    type_principal,
    price_level,
    is_active
  );

CREATE INDEX IF NOT EXISTS idx_gold_mv_datatourisme_prime_serving_score
  ON gold.mv_datatourisme_prime_serving (score_prime DESC);

-- Optional refresh:
-- REFRESH MATERIALIZED VIEW CONCURRENTLY gold.mv_datatourisme_prime_serving;
