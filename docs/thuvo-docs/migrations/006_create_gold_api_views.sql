-- =====================================================
-- GOLD API VIEW : clean contract for frontend / API
-- =====================================================

CREATE OR REPLACE VIEW gold.v_api_poi_prime AS
SELECT
    s.poi_id AS poi_id,

    -- identité
    p.name,
    p.main_category_compressed        AS category,

    -- typologies
    p.format_label                    AS format,
    p.tempo_label                     AS tempo,

    -- scoring
    s.final_score,

    -- metadata
    s.computed_at

FROM silver.prime_classique p
JOIN gold.poi_score s
  ON s.poi_id = p.source_id;
