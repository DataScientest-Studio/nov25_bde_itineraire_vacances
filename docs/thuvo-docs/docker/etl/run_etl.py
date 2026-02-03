import os
import pandas as pd
from sqlalchemy import create_engine, text

# --- Connexion DB ---
DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --- Fichiers montés dans le container via docker-compose ---
prime_classique_path = os.environ["PRIME_CLASSIQUE_PATH"]

print("Reading parquet:", prime_classique_path)
df = pd.read_parquet(prime_classique_path)

print("Rows:", len(df))
print("First columns:", list(df.columns)[:15])

# Exemple: charge juste un échantillon pour valider Docker + DB
sample = df.head(500).copy()
sample.columns = [c.lower().strip().replace(" ", "_") for c in sample.columns]

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))

sample.to_sql("prime_sample", engine, schema="bronze", if_exists="replace", index=False)

print("Loaded into Postgres: bronze.prime_sample (replace)")
