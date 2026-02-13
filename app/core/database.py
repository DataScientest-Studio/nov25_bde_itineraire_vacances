from dotenv import load_dotenv
from psycopg2 import pool
import os

class DBManager:
    def __init__(self):
        #load_dotenv()
        self.pool = pool.SimpleConnectionPool(
            minconn=5,
            maxconn=10,
            host=os.getenv("POSTGRES_VACANCES_HOST", "postgres-vacances"),
            port=int(os.getenv("POSTGRES_VACANCES_PORT", 5432)),
            database=os.getenv("POSTGRES_VACANCES_DB"),
            user=os.getenv("POSTGRES_VACANCES_USER"),
            password=os.getenv("POSTGRES_VACANCES_PASSWORD")
        )

    def get_conn(self):
        return self.pool.getconn()

    def return_conn(self, conn):
        self.pool.putconn(conn)

db_manager = DBManager()