from dotenv import load_dotenv
from psycopg2 import pool
import os

class DBManager:
    def __init__(self):
        load_dotenv()
        self.pool = pool.SimpleConnectionPool(
            minconn=5,
            maxconn=10,
            database=os.getenv("POSTGRES_DB"),
            host=os.getenv("DB_HOST"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("DB_PORT"),
        )

    def get_conn(self):
        return self.pool.getconn()

    def return_conn(self, conn):
        self.pool.putconn(conn)

db_manager = DBManager()