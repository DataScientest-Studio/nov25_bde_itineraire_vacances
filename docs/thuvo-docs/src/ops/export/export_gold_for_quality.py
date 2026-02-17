import os
import pandas as pd
import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "prime")
DB_USER = os.getenv("POSTGRES_USER", "prime")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "prime")

DSN = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)


def export_query(sql: str, outfile: str):
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(sql, conn)
    df.to_parquet(outfile, index=False)
    print("exported:", outfile, "rows=", len(df))


def main():
    # PRIME = dataset métier final
    export_query(
        "SELECT * FROM gold.v_prime_classique_api",
        f"{OUT_DIR}/gold_prime.parquet"
    )

    # TRIPADVISOR = dataset discovery
    export_query(
        "SELECT * FROM gold.mv_tripadvisor_discover",
        f"{OUT_DIR}/gold_tripadvisor.parquet"
    )


if __name__ == "__main__":
    main()
