-- 044_silver_datatourisme_prime_quality_checks.sql
-- Quality checks "hard fail" (raise exception) for silver.prime_classique
-- Aligned with 041_silver_datatourisme_prime.sql + 042 indexes + constraints decisions
-- (no checks for max_people positivity; no enum check for price_level)

CREATE SCHEMA IF NOT EXISTS ops;

CREATE OR REPLACE FUNCTION ops.quality_check_datatourisme_prime(
  -- SELECT ops.quality_check_datatourisme_prime();
  p_min_rows bigint DEFAULT 1000,
  p_min_rating numeric DEFAULT 0,
  p_max_rating numeric DEFAULT 5,
  p_min_reviews bigint DEFAULT 0,
  p_max_distance_km numeric DEFAULT 5000, -- guardrail anti-bug
  p_max_score numeric DEFAULT 1000000,  -- guardrail anti-bug
  p_max_weight numeric DEFAULT 1000     -- guardrail anti-bug
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  v_total bigint;

  v_nulls_key bigint;
  v_blank_source_id bigint;
  v_blank_city bigint;
  v_blank_main_category bigint;

  v_bad_lat bigint;
  v_bad_lon bigint;
  v_bad_rating bigint;

  v_bad_reviews bigint;

  v_bad_distance_low bigint;
  v_bad_distance_high bigint;

  v_bad_score_low bigint;
  v_bad_score_high bigint;

  v_bad_main_cat_weight_low bigint;
  v_bad_main_cat_weight_high bigint;

  v_bad_format_weight_low bigint;
  v_bad_format_weight_high bigint;

  v_bad_tempo_weight_low bigint;
  v_bad_tempo_weight_high bigint;

  v_missing_geom bigint;
  v_dupe_source_id bigint;

  v_content_hash_null bigint;
BEGIN

  -------------------------------------------------------
  -- 1) Volume minimal
  -------------------------------------------------------
  SELECT COUNT(*) INTO v_total
  FROM silver.prime_classique;

  IF v_total < p_min_rows THEN
    RAISE EXCEPTION '[QC datatourisme_prime] Volume trop faible: %, attendu >= %', v_total, p_min_rows;
  END IF;

  -------------------------------------------------------
  -- 2) Nulls / blanks sur colonnes clés
  -- (clés indispensables pour UI + carte + scoring)
  -------------------------------------------------------
  SELECT COUNT(*) INTO v_nulls_key
  FROM silver.prime_classique
  WHERE source_id IS NULL
     OR lat IS NULL
     OR lon IS NULL
     OR main_category IS NULL
     OR score_prime IS NULL
     OR city IS NULL;

  IF v_nulls_key > 0 THEN
    RAISE EXCEPTION
      '[QC datatourisme_prime] Nulls sur colonnes clés (source_id/lat/lon/main_category/score_prime/city): %',
      v_nulls_key;
  END IF;

  SELECT COUNT(*) INTO v_blank_source_id
  FROM silver.prime_classique
  WHERE source_id IS NOT NULL AND btrim(source_id) = '';

  IF v_blank_source_id > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] source_id vide: %', v_blank_source_id;
  END IF;

  SELECT COUNT(*) INTO v_blank_city
  FROM silver.prime_classique
  WHERE city IS NOT NULL AND btrim(city) = '';

  IF v_blank_city > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] city vide: %', v_blank_city;
  END IF;

  SELECT COUNT(*) INTO v_blank_main_category
  FROM silver.prime_classique
  WHERE main_category IS NOT NULL AND btrim(main_category) = '';

  IF v_blank_main_category > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] main_category vide: %', v_blank_main_category;
  END IF;

  -------------------------------------------------------
  -- 3) Règles de cohérence (valeurs)
  -------------------------------------------------------
  -- lat/lon ranges
  SELECT COUNT(*) INTO v_bad_lat
  FROM silver.prime_classique
  WHERE lat < -90 OR lat > 90;

  IF v_bad_lat > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] lat hors [-90..90]: %', v_bad_lat;
  END IF;

  SELECT COUNT(*) INTO v_bad_lon
  FROM silver.prime_classique
  WHERE lon < -180 OR lon > 180;

  IF v_bad_lon > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] lon hors [-180..180]: %', v_bad_lon;
  END IF;

  -- rating in [0,5]
  SELECT COUNT(*) INTO v_bad_rating
  FROM silver.prime_classique
  WHERE rating IS NOT NULL
    AND (rating < p_min_rating OR rating > p_max_rating);

  IF v_bad_rating > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] rating hors [%..%]: %', p_min_rating, p_max_rating, v_bad_rating;
  END IF;

  -- review_count non-negative (column exists; often placeholder -> NULL ok)
  SELECT COUNT(*) INTO v_bad_reviews
  FROM silver.prime_classique
  WHERE review_count IS NOT NULL
    AND review_count < p_min_reviews;

  IF v_bad_reviews > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] review_count < %: %', p_min_reviews, v_bad_reviews;
  END IF;


  -- distance_km >= 0 and < guardrail (often placeholder -> NULL ok)
  SELECT COUNT(*) INTO v_bad_distance_low
  FROM silver.prime_classique
  WHERE distance_km IS NOT NULL AND distance_km < 0;

  IF v_bad_distance_low > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] distance_km < 0: %', v_bad_distance_low;
  END IF;

  SELECT COUNT(*) INTO v_bad_distance_high
  FROM silver.prime_classique
  WHERE distance_km IS NOT NULL AND distance_km >= p_max_distance_km;

  IF v_bad_distance_high > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] distance_km >= % (suspect): %', p_max_distance_km, v_bad_distance_high;
  END IF;

  -- score_prime sanity
  SELECT COUNT(*) INTO v_bad_score_low
  FROM silver.prime_classique
  WHERE score_prime IS NOT NULL AND score_prime < 0;

  IF v_bad_score_low > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] score_prime < 0: %', v_bad_score_low;
  END IF;

  SELECT COUNT(*) INTO v_bad_score_high
  FROM silver.prime_classique
  WHERE score_prime IS NOT NULL AND score_prime >= p_max_score;

  IF v_bad_score_high > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] score_prime >= % (suspect): %', p_max_score, v_bad_score_high;
  END IF;

  -- weights non-negative + guardrail (anti-bug)
  SELECT COUNT(*) INTO v_bad_main_cat_weight_low
  FROM silver.prime_classique
  WHERE main_cat_weight IS NOT NULL AND main_cat_weight < 0;

  IF v_bad_main_cat_weight_low > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] main_cat_weight < 0: %', v_bad_main_cat_weight_low;
  END IF;

  SELECT COUNT(*) INTO v_bad_main_cat_weight_high
  FROM silver.prime_classique
  WHERE main_cat_weight IS NOT NULL AND main_cat_weight >= p_max_weight;

  IF v_bad_main_cat_weight_high > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] main_cat_weight >= % (suspect): %', p_max_weight, v_bad_main_cat_weight_high;
  END IF;

  SELECT COUNT(*) INTO v_bad_format_weight_low
  FROM silver.prime_classique
  WHERE format_weight IS NOT NULL AND format_weight < 0;

  IF v_bad_format_weight_low > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] format_weight < 0: %', v_bad_format_weight_low;
  END IF;

  SELECT COUNT(*) INTO v_bad_format_weight_high
  FROM silver.prime_classique
  WHERE format_weight IS NOT NULL AND format_weight >= p_max_weight;

  IF v_bad_format_weight_high > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] format_weight >= % (suspect): %', p_max_weight, v_bad_format_weight_high;
  END IF;

  SELECT COUNT(*) INTO v_bad_tempo_weight_low
  FROM silver.prime_classique
  WHERE tempo_weight IS NOT NULL AND tempo_weight < 0;

  IF v_bad_tempo_weight_low > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] tempo_weight < 0: %', v_bad_tempo_weight_low;
  END IF;

  SELECT COUNT(*) INTO v_bad_tempo_weight_high
  FROM silver.prime_classique
  WHERE tempo_weight IS NOT NULL AND tempo_weight >= p_max_weight;

  IF v_bad_tempo_weight_high > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] tempo_weight >= % (suspect): %', p_max_weight, v_bad_tempo_weight_high;
  END IF;

  -------------------------------------------------------
  -- 4) Sanity checks techniques
  -------------------------------------------------------

  -- geom devrait être présent si lat/lon présents
  SELECT COUNT(*) INTO v_missing_geom
  FROM silver.prime_classique
  WHERE lat IS NOT NULL AND lon IS NOT NULL AND geom IS NULL;

  IF v_missing_geom > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] geom NULL alors que lat/lon présents: %', v_missing_geom;
  END IF;

  -- Unicité logique de source_id (PK existe, mais check utile si données chargées ailleurs / vues)
  SELECT COUNT(*) INTO v_dupe_source_id
  FROM (
    SELECT source_id
    FROM silver.prime_classique
    GROUP BY source_id
    HAVING COUNT(*) > 1
  ) t;

  IF v_dupe_source_id > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] source_id dupliqués: %', v_dupe_source_id;
  END IF;

  -- content_hash should be populated in SILVER (SCD2 / idempotence)
  SELECT COUNT(*) INTO v_content_hash_null
  FROM silver.prime_classique
  WHERE content_hash IS NULL OR btrim(content_hash) = '';

  IF v_content_hash_null > 0 THEN
    RAISE EXCEPTION '[QC datatourisme_prime] content_hash NULL/blank: % (SILVER should compute it)', v_content_hash_null;
  END IF;

  -- OK
  RAISE NOTICE '[QC datatourisme_prime] OK (rows=%)', v_total;
END;
$$;

-- Exécution (manuelle / CI)
-- SELECT ops.quality_check_datatourisme_prime();
