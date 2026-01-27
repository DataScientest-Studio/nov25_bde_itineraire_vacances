-- ============================================================
-- MIGRATION 005
-- GOLD VIEWS : API / APPLICATION
-- ============================================================

CREATE SCHEMA IF NOT EXISTS gold;

-- ------------------------------------------------------------
-- Vue principale : POI + score
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW gold.v_poi_scored AS
SELECT
  p.*,
  s.final_score,
  s.computed_at
FROM silver.prime_classique p
JOIN gold.poi_score s
  ON s.poi_id = p.source_id;

-- ------------------------------------------------------------
-- Vue classement
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW gold.v_poi_top AS
SELECT
  poi_id,
  final_score,
  computed_at
FROM gold.poi_score
ORDER BY final_score DESC;

-- ------------------------------------------------------------
-- Vue agrégée par catégorie
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW gold.v_score_by_category AS
SELECT
  p.main_category_compressed,
  COUNT(*) AS n_poi,
  AVG(s.final_score) AS avg_score,
  MAX(s.final_score) AS max_score
FROM silver.prime_classique p
JOIN gold.poi_score s
  ON s.poi_id = p.source_id
GROUP BY 1
ORDER BY avg_score DESC;
