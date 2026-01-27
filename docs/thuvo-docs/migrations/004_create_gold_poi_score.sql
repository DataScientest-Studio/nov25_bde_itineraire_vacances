-- =====================================================
-- MIGRATION 004
-- GOLD LAYER : POI FINAL SCORE
-- FINAL_SCORE = main_cat_weight * (1 + format_weight + tempo_weight)
-- =====================================================

CREATE SCHEMA IF NOT EXISTS gold;

DROP TABLE IF EXISTS gold.poi_score;

CREATE TABLE gold.poi_score AS
SELECT
    -- id normalisé API
    source_id AS poi_id,

    -- dimensions API
    main_category_compressed AS main_category,
    format_label            AS format,
    tempo_label             AS tempo,

    -- weights
    main_cat_weight,
    format_weight,
    tempo_weight,

    -- final score
    main_cat_weight * (1 + format_weight + tempo_weight) AS final_score,

    -- metadata
    CURRENT_TIMESTAMP AS computed_at
FROM silver.prime_classique;

-- Indexes for API performance
CREATE INDEX IF NOT EXISTS idx_gold_poi_score_poi_id
    ON gold.poi_score(poi_id);

CREATE INDEX IF NOT EXISTS idx_gold_poi_score_score_desc
    ON gold.poi_score(final_score DESC);

-- CHECK
SELECT COUNT(*) FROM gold.poi_score;
