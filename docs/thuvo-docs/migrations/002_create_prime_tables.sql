-- ============================================================
-- 002_create_prime_tables.sql  (CORRIGÉ)
-- Tables PRIME (Datatourisme flippé) : classique + excursion
-- + historisation SCD2 light
--
-- Points corrigés :
-- - postal_code = TEXT (car valeurs possibles "XXXX Cedex", etc.)
-- - cohérence des types entre table courante et table history
--   (region_insee / dept_insee / city_insee restent INTEGER)
-- - geom est généré depuis (lon, lat) au moment du LOAD
-- ============================================================

-- Assure que PostGIS est dispo (si déjà fait ailleurs, pas grave)
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- 1) Tables "courantes" (snapshot)
-- ============================================================

CREATE TABLE IF NOT EXISTS silver.prime_classique (
  source_id TEXT PRIMARY KEY,

  poi_types TEXT,
  name TEXT,
  label_en TEXT,
  snippet TEXT,
  snippet_en TEXT,

  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  geom geometry(Point, 4326),

  country TEXT,
  region_insee INTEGER,
  region TEXT,
  dept_insee INTEGER,
  departement TEXT,
  city_insee INTEGER,
  city TEXT,

  -- IMPORTANT : TEXT (CEDEX, etc.)
  postal_code TEXT,

  address_locality TEXT,
  address TEXT,

  theme_fr TEXT,
  theme_en TEXT,
  architectural_style_fr TEXT,
  architectural_style_en TEXT,

  main_media_url TEXT,
  media_resource_url TEXT,
  url TEXT,

  rating DOUBLE PRECISION,
  review_value_label_fr TEXT,
  review_value_label_en TEXT,
  sub_category TEXT,

  opens_time TEXT,
  closes_time TEXT,
  hours_valid_from TEXT,
  hours_valid_through TEXT,

  max_people INTEGER,

  difficulty_level_fr TEXT,
  locomotion_mode_fr TEXT,
  tour_type_fr TEXT,

  tour_distance_m DOUBLE PRECISION,
  duration_min DOUBLE PRECISION,
  practice_duration_min DOUBLE PRECISION,
  duration_days DOUBLE PRECISION,
  practice_duration_days DOUBLE PRECISION,
  positive_elevation_gain_m DOUBLE PRECISION,
  negative_elevation_loss_m DOUBLE PRECISION,

  start_date TEXT,
  end_date TEXT,
  last_update_datatourisme TEXT,

  price DOUBLE PRECISION,
  min_price DOUBLE PRECISION,
  max_price DOUBLE PRECISION,

  type_principal TEXT,

  is_resto_type BOOLEAN,
  is_resto_label BOOLEAN,
  is_resto BOOLEAN,

  is_label_incontournable BOOLEAN,
  is_label_famille BOOLEAN,
  is_label_handicap BOOLEAN,
  is_label_hebergement BOOLEAN,
  is_label_gastronomie BOOLEAN,
  is_label_artisanat BOOLEAN,
  is_label_randonnee BOOLEAN,
  is_label_green BOOLEAN,
  is_etoile BOOLEAN,

  price_level TEXT,
  main_category_compressed TEXT,
  is_prime_plus BOOLEAN,

  main_cat_weight DOUBLE PRECISION,
  main_cat_candidates TEXT,

  format_label TEXT,
  format_weight DOUBLE PRECISION,
  tempo_label TEXT,
  tempo_weight DOUBLE PRECISION,

  score_prime DOUBLE PRECISION,
  source TEXT,

  review_count INTEGER,
  distance_km DOUBLE PRECISION,

  content_hash TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prime_classique_geom
ON silver.prime_classique USING GIST(geom);

-- Table excursion = clone de la table classique (mêmes colonnes/types/indexes/contraintes)
CREATE TABLE IF NOT EXISTS silver.prime_excursion (
  LIKE silver.prime_classique INCLUDING ALL
);

CREATE INDEX IF NOT EXISTS idx_prime_excursion_geom
ON silver.prime_excursion USING GIST(geom);

-- ============================================================
-- 2) Tables historiques SCD2 light
-- ============================================================

CREATE TABLE IF NOT EXISTS silver.prime_classique_history (
  history_id BIGSERIAL PRIMARY KEY,
  source_id TEXT NOT NULL,

  valid_from TIMESTAMPTZ NOT NULL,
  valid_to TIMESTAMPTZ,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,

  -- copie des colonnes business (mêmes types que la table courante)
  poi_types TEXT,
  name TEXT,
  label_en TEXT,
  snippet TEXT,
  snippet_en TEXT,

  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  geom geometry(Point, 4326),

  country TEXT,
  region_insee INTEGER,
  region TEXT,
  dept_insee INTEGER,
  departement TEXT,
  city_insee INTEGER,
  city TEXT,

  -- IMPORTANT : TEXT (CEDEX, etc.)
  postal_code TEXT,

  address_locality TEXT,
  address TEXT,

  theme_fr TEXT,
  theme_en TEXT,
  architectural_style_fr TEXT,
  architectural_style_en TEXT,

  main_media_url TEXT,
  media_resource_url TEXT,
  url TEXT,

  rating DOUBLE PRECISION,
  review_value_label_fr TEXT,
  review_value_label_en TEXT,
  sub_category TEXT,

  opens_time TEXT,
  closes_time TEXT,
  hours_valid_from TEXT,
  hours_valid_through TEXT,

  max_people INTEGER,

  difficulty_level_fr TEXT,
  locomotion_mode_fr TEXT,
  tour_type_fr TEXT,

  tour_distance_m DOUBLE PRECISION,
  duration_min DOUBLE PRECISION,
  practice_duration_min DOUBLE PRECISION,
  duration_days DOUBLE PRECISION,
  practice_duration_days DOUBLE PRECISION,
  positive_elevation_gain_m DOUBLE PRECISION,
  negative_elevation_loss_m DOUBLE PRECISION,

  start_date TEXT,
  end_date TEXT,
  last_update_datatourisme TEXT,

  price DOUBLE PRECISION,
  min_price DOUBLE PRECISION,
  max_price DOUBLE PRECISION,

  type_principal TEXT,

  is_resto_type BOOLEAN,
  is_resto_label BOOLEAN,
  is_resto BOOLEAN,

  is_label_incontournable BOOLEAN,
  is_label_famille BOOLEAN,
  is_label_handicap BOOLEAN,
  is_label_hebergement BOOLEAN,
  is_label_gastronomie BOOLEAN,
  is_label_artisanat BOOLEAN,
  is_label_randonnee BOOLEAN,
  is_label_green BOOLEAN,
  is_etoile BOOLEAN,

  price_level TEXT,
  main_category_compressed TEXT,
  is_prime_plus BOOLEAN,

  main_cat_weight DOUBLE PRECISION,
  main_cat_candidates TEXT,

  format_label TEXT,
  format_weight DOUBLE PRECISION,
  tempo_label TEXT,
  tempo_weight DOUBLE PRECISION,

  score_prime DOUBLE PRECISION,
  source TEXT,

  review_count INTEGER,
  distance_km DOUBLE PRECISION,

  content_hash TEXT NOT NULL
);

-- Index pour récupérer vite la version active
CREATE INDEX IF NOT EXISTS idx_prime_classique_hist_active
ON silver.prime_classique_history(source_id)
WHERE is_active;

-- Index spatial pour audits / analyses
CREATE INDEX IF NOT EXISTS idx_prime_classique_hist_geom
ON silver.prime_classique_history USING GIST(geom);

-- Table excursion history = clone de la history classique
CREATE TABLE IF NOT EXISTS silver.prime_excursion_history (
  LIKE silver.prime_classique_history INCLUDING ALL
);

CREATE INDEX IF NOT EXISTS idx_prime_excursion_hist_active
ON silver.prime_excursion_history(source_id)
WHERE is_active;

CREATE INDEX IF NOT EXISTS idx_prime_excursion_hist_geom
ON silver.prime_excursion_history USING GIST(geom);
