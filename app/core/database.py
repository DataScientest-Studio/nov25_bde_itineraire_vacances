from dotenv import load_dotenv
from psycopg2 import pool
import os

class DBManager:
    def __init__(self):
        load_dotenv()
        self.pool = pool.SimpleConnectionPool(
            minconn=5,
            maxconn=10,
            database=os.getenv("POSTGRES_DB", "vacances"),
            host=os.getenv("DB_HOST", "postgres-vacances"),
            user=os.getenv("POSTGRES_USER", "vacances_user"),
            password=os.getenv("POSTGRES_PASSWORD", "vacances_password"),
            port=int(os.getenv("DB_PORT", "5432")),
        )

    def get_conn(self):
        return self.pool.getconn()

    def return_conn(self, conn):
        self.pool.putconn(conn)

db_manager = DBManager()