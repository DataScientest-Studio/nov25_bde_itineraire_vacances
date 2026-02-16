-- 044_silver_tripadvisor_quality_checks.sql
-- Quality checks "hard fail" pour silver.tripadvisor_france
-- Objectif : sécuriser la couche SILVER (PK, bornes, cohérence géo)

CREATE SCHEMA IF NOT EXISTS ops;

CREATE OR REPLACE FUNCTION ops.quality_check_tripadvisor_france(
  p_min_rows bigint DEFAULT 1000,

  -- Géo
  p_max_null_geo_ratio numeric DEFAULT 0.05,   -- tolérance 5%
  p_min_lat numeric DEFAULT -90,
  p_max_lat numeric DEFAULT 90,
  p_min_lon numeric DEFAULT -180,
  p_max_lon numeric DEFAULT 180,

  -- Optionnel : si tu veux imposer content_hash (SCD2/idempotence)
  p_require_content_hash boolean DEFAULT TRUE
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  v_total bigint;
  v_dupe_pk bigint;
  v_null_pk bigint;

  v_null_latlon bigint;
  v_null_latlon_ratio numeric;

  v_bad_lat bigint;
  v_bad_lon bigint;

  v_bad_geom bigint;
  v_bad_content_hash bigint;
BEGIN
  -------------------------------------------------------
  -- 1) Volume minimal
  -------------------------------------------------------
  SELECT COUNT(*) INTO v_total
  FROM silver.tripadvisor_france;

  IF v_total < p_min_rows THEN
    RAISE EXCEPTION '[QC tripadvisor_france] volume trop faible (%, attendu >= %)', v_total, p_min_rows;
  END IF;

  -------------------------------------------------------
  -- 2) PK non NULL + pas de doublons
  -------------------------------------------------------
  SELECT COUNT(*) INTO v_null_pk
  FROM silver.tripadvisor_france
  WHERE source_id IS NULL OR btrim(source_id) = '';

  IF v_null_pk > 0 THEN
    RAISE EXCEPTION '[QC tripadvisor_france] source_id NULL/empty: %', v_null_pk;
  END IF;

  SELECT COUNT(*) INTO v_dupe_pk
  FROM (
    SELECT source_id
    FROM silver.tripadvisor_france
    GROUP BY source_id
    HAVING COUNT(*) > 1
  ) d;

  IF v_dupe_pk > 0 THEN
    RAISE EXCEPTION '[QC tripadvisor_france] doublons de source_id: %', v_dupe_pk;
  END IF;

  -------------------------------------------------------
  -- 3) Géo : taux de NULL + bornes + cohérence geom
  -------------------------------------------------------
  SELECT COUNT(*) INTO v_null_latlon
  FROM silver.tripadvisor_france
  WHERE lat IS NULL OR lon IS NULL;

  v_null_latlon_ratio :=
    CASE WHEN v_total = 0 THEN 1
         ELSE (v_null_latlon::numeric / v_total::numeric)
    END;

  IF v_null_latlon_ratio > p_max_null_geo_ratio THEN
    RAISE EXCEPTION '[QC tripadvisor_france] trop de lat/lon NULL (ratio=%, max=%)',
      v_null_latlon_ratio, p_max_null_geo_ratio;
  END IF;

  SELECT COUNT(*) INTO v_bad_lat
  FROM silver.tripadvisor_france
  WHERE lat IS NOT NULL AND (lat < p_min_lat OR lat > p_max_lat);

  IF v_bad_lat > 0 THEN
    RAISE EXCEPTION '[QC tripadvisor_france] lat hors bornes: %', v_bad_lat;
  END IF;

  SELECT COUNT(*) INTO v_bad_lon
  FROM silver.tripadvisor_france
  WHERE lon IS NOT NULL AND (lon < p_min_lon OR lon > p_max_lon);

  IF v_bad_lon > 0 THEN
    RAISE EXCEPTION '[QC tripadvisor_france] lon hors bornes: %', v_bad_lon;
  END IF;

  -- Si lat/lon présents => geom doit exister
  SELECT COUNT(*) INTO v_bad_geom
  FROM silver.tripadvisor_france
  WHERE lat IS NOT NULL AND lon IS NOT NULL AND geom IS NULL;

  IF v_bad_geom > 0 THEN
    RAISE EXCEPTION '[QC tripadvisor_france] geom manquante alors que lat/lon présents: %', v_bad_geom;
  END IF;

  -------------------------------------------------------
  -- 4) content_hash (optionnel)
  -------------------------------------------------------
  IF p_require_content_hash THEN
    SELECT COUNT(*) INTO v_bad_content_hash
    FROM silver.tripadvisor_france
    WHERE content_hash IS NULL OR btrim(content_hash) = '';

    IF v_bad_content_hash > 0 THEN
      RAISE EXCEPTION '[QC tripadvisor_france] content_hash NULL/blank: %', v_bad_content_hash;
    END IF;
  END IF;

  -- OK
  RAISE NOTICE '[QC tripadvisor_france] OK (rows=%, null_geo_ratio=%)',
    v_total, v_null_latlon_ratio;
END;
$$;

-- Exécution (manuel / CI)
-- SELECT ops.quality_check_tripadvisor_france();
