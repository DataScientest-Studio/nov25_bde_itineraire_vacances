import os

from dotenv import load_dotenv
from psycopg2 import pool


class DBManager:
    def __init__(self, db_env):
        load_dotenv(db_env)

        # création d'un pool de connecteurs à la BDD :
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
        # récupérer une connection du pool de connection
        return self.pool.getconn()

    def return_conn(self, conn):
        # remettre la connection dans le pool
        self.pool.putconn(conn)

    def execute_query(self, query, params=None):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            result = cur.fetchall()
            return result
        finally:
            cur.close()
            self.return_conn(conn)

    def get_main_categories(self):
        query = """ SELECT DISTINCT(nom_cat)
                    FROM main_category
                    ORDER BY nom_cat;"""
        result = self.execute_query(query)
        main_categories_list = [row[0] for row in result]
        return main_categories_list

    def get_sub_categories(self, main_categories_list):
        query = """ SELECT DISTINCT(nom_sous_cat)
                    FROM sub_category
                    WHERE main_category_id IN (SELECT id FROM main_category WHERE nom_cat IN %s )
                    ORDER BY nom_sous_cat;"""
        result = self.execute_query(query, (tuple(main_categories_list),))
        categories_list = [row[0] for row in result]
        return categories_list

    def search_poi(self, longitude, latitude, radius, sub_categories):
        query = """
                    SELECT DISTINCT
                        p.poi_id,
                        p.longitude,
                        p.latitude, 
                        c.sub_category
                    FROM pois AS p
                    LEFT JOIN poi_category as pc USING(poi_id)
                    LEFT JOIN categories AS c USING(category_id)
                    WHERE (ST_DWithin(p.geom, 
                                    ST_SetSRID(ST_Point(%s, %s), 4326),
                                    %s))
                           AND 
                           (c.sub_category IN %s)
                    ;"""

        result = self.execute_query(
            query, (longitude, latitude, radius, tuple(sub_categories))
        )
        return result

    def get_poi_data(self, poi_id):
        query = """
                    SELECT DISTINCT
                        p.poi_id,
                        p.label,
                        p.description,
                        p.latitude,
                        p.longitude,
                        a.street_address,
                        a.postal_code,
                        a.locality,
                        ph.phone_num,
                        m.mail_address
                    FROM pois AS p
                    LEFT JOIN poi_category as pc USING(poi_id)
                    LEFT JOIN categories AS c USING(category_id)
                    LEFT JOIN addresses AS a USING(address_id)
                    LEFT JOIN phone_contact AS ph USING(poi_id)
                    LEFT JOIN mail_contact AS m USING(poi_id)
                    WHERE (p.poi_id = %s)
                    ;"""

        result = self.execute_query(query, (poi_id,))
        return result[0]
