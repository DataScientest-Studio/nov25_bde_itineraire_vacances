-- 022_raw_datatourisme_prime.sql
-- RAW layer for DataTourisme Prime
-- Mirror of parquet structure (no business logic)

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.datatourisme_prime (

  -- Identité / textes
  source_id TEXT,
  name TEXT,
  label_en TEXT,
  snippet TEXT,
  snippet_en TEXT,

  -- Géoloc
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,

  -- Localisation admin
  country TEXT,
  region_insee TEXT,
  region TEXT,
  dept_insee TEXT,
  departement TEXT,
  city_insee TEXT,
  city TEXT,
  postal_code TEXT,
  address_locality TEXT,
  address TEXT,

  -- Thèmes
  theme_fr TEXT,
  theme_en TEXT,

  -- Média / url
  media_resource_url TEXT,
  url TEXT,

  -- Avis
  rating DOUBLE PRECISION,
  review_value_label_fr TEXT,
  review_value_label_en TEXT,
  sub_category TEXT,

  -- Capacités / itinéraires
  max_people INTEGER,
  difficulty_level_fr TEXT,
  locomotion_mode_fr TEXT,
  tour_type_fr TEXT,
  last_update_datatourisme TIMESTAMPTZ,

  -- Prix
  price DOUBLE PRECISION,
  price_level TEXT,
  price_range TEXT,

  -- Flags labels (string volontairement en RAW)
  is_label_incontournable TEXT,
  is_label_handicap TEXT,
  is_label_hebergement TEXT,
  is_label_gastronomy TEXT,
  is_label_artisanat TEXT,
  is_label_itinerary TEXT,
  is_label_green TEXT,

  -- Prime enrichi (encore brut ici)
  main_category TEXT,
  main_cat_weight DOUBLE PRECISION,
  type_principal TEXT,
  format_weight DOUBLE PRECISION,
  tempo_weight DOUBLE PRECISION,
  score_prime DOUBLE PRECISION,
  hebergement_type TEXT,

  -- Buckets / saison / horaires / relief
  tour_distance_bucket TEXT,
  practice_duration_final_bucket TEXT,
  seasons_available TEXT,
  opening_hours_str TEXT,
  elevation_gain_loss TEXT,

  -- Source
  source TEXT,

  -- placeholders UI
  review_count INTEGER,
  distance_km DOUBLE PRECISION,

  -- Meta ingestion
  raw_loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  raw_source_file TEXT
);

-- Index technique ingestion
CREATE INDEX IF NOT EXISTS idx_raw_datatourisme_prime_source_id
ON raw.datatourisme_prime (source_id);
