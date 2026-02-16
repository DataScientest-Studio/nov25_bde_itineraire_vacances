-- 051_gold_tripadvisor.sql
-- GOLD layer - TripAdvisor (serving MV)
-- Compatible with current schema: silver.tripadvisor_france (no is_active column)

CREATE SCHEMA IF NOT EXISTS gold;

DROP MATERIALIZED VIEW IF EXISTS gold.mv_tripadvisor_discover;

CREATE MATERIALIZED VIEW gold.mv_tripadvisor_discover AS
SELECT
  source_id,

  -- keep name "type" for UI contract
  type AS "type",

  lat::double precision AS lat,
  lon::double precision AS lon,

  city,

  rating::double precision AS rating,
  review_count,

  -- normalize price_level (soft)
  CASE
    WHEN price_level IS NULL THEN NULL
    WHEN lower(price_level::text) IN ('eco','normal','confort','premium')
      THEN lower(price_level::text)
    ELSE NULL
  END AS price_level,

  reco_score::double precision AS reco_score,

  cuisine_continent,

  COALESCE(vegetarian_friendly, FALSE) AS vegetarian_friendly,
  COALESCE(vegan_options, FALSE)       AS vegan_options,
  COALESCE(gluten_free, FALSE)         AS gluten_free,
  COALESCE(is_halal, FALSE)            AS is_halal,
  COALESCE(is_kosher, FALSE)           AS is_kosher,

  geom,

  TRUE::boolean AS is_active,
  COALESCE(updated_at, NOW()) AS serving_timestamp

FROM silver.tripadvisor_france
WHERE geom IS NOT NULL
  AND lat IS NOT NULL
  AND lon IS NOT NULL;

-- Unique index required for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS ux_gold_tripadvisor_discover_source_id
  ON gold.mv_tripadvisor_discover (source_id);

-- Filters index (read side)
CREATE INDEX IF NOT EXISTS idx_gold_tripadvisor_discover_filters
  ON gold.mv_tripadvisor_discover (
    city,
    "type",
    price_level,
    rating,
    review_count,
    reco_score,
    cuisine_continent,
    vegetarian_friendly,
    vegan_options,
    gluten_free,
    is_halal,
    is_kosher,
    is_active
  );

-- Spatial index
CREATE INDEX IF NOT EXISTS idx_gold_tripadvisor_discover_geom
  ON gold.mv_tripadvisor_discover USING GIST (geom);

-- Optional refresh:
-- REFRESH MATERIALIZED VIEW CONCURRENTLY gold.mv_tripadvisor_discover;
