-- 003_fix_prime_postal_code_text.sql
-- Prime: postal_code peut contenir "CEDEX" -> doit être TEXT

ALTER TABLE silver.prime_classique
  ALTER COLUMN postal_code TYPE TEXT USING postal_code::text;

ALTER TABLE silver.prime_excursion
  ALTER COLUMN postal_code TYPE TEXT USING postal_code::text;

ALTER TABLE silver.prime_classique_history
  ALTER COLUMN postal_code TYPE TEXT USING postal_code::text;

ALTER TABLE silver.prime_excursion_history
  ALTER COLUMN postal_code TYPE TEXT USING postal_code::text;
