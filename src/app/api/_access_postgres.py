import psycopg2

def test_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",      # 👈 important
            port=5433,             # 👈 important
            user="vacances_user",
            password="vacances_password",
            dbname="vacances"
        )

        print("Connexion réussie !")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            print("Tables :", tables)

        conn.close()

    except Exception as e:
        print("Erreur :", e)

if __name__ == "__main__":
    test_connection()