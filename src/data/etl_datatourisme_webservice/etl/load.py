import psycopg2
from dotenv import load_dotenv
import os


class DBManager:
    def __init__(self, db_env) :
        
        load_dotenv(db_env)
    
        # création du connecteur à la BDD :
        self.conn = psycopg2.connect(
                        database= os.getenv('POSTGRES_DB'),
                        host= os.getenv('DB_HOST'),
                        user= os.getenv('POSTGRES_USER'),
                        password=  os.getenv('POSTGRES_PASSWORD'),
                        port= os.getenv('DB_PORT'))

        # création du curseur :
        self.cur = self.conn.cursor()
    
    def create_tables(self) :

        # activities table :
        self.cur.execute(""" 
                         CREATE TABLE IF NOT EXISTS activities (
                            activity_id serial PRIMARY KEY,
                            activity_duration INT NOT NULL);
                         """)
        
        # categories table :
        self.cur.execute(""" 
                         CREATE TABLE IF NOT EXISTS categories (
                            category_id serial PRIMARY KEY,
                            main_category VARCHAR(45) NOT NULL,
                            sub_category VARCHAR(45) NOT NULL,
                            itinerary_category BOOLEAN NOT NULL);
                         """)
        
        # localities table
        self.cur.execute(""" 
                         CREATE TABLE  IF NOT EXISTS localities (
                            locality_id serial PRIMARY KEY,
                            locality_name VARCHAR(45) NOT NULL,
                            locality_dep_name VARCHAR(45) NOT NULL,
                            locality_dep_code INT NOT NULL,
                            locality_reg_name VARCHAR(45) NOT NULL);
                        """)
        # h3 table
        self.cur.execute(""" 
                         CREATE TABLE IF NOT EXISTS h3_levels (
                            h3_5_id VARCHAR(45) PRIMARY KEY,
                            h3_7_id VARCHAR(45) NOT NULL,
                            h3_9_id VARCHAR(45) NOT NULL);
                        """)
        
        # addresses table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS addresses (
                            address_id serial PRIMARY KEY,
                            street_address VARCHAR(255),
                            postal_code INT NOT NULL,
                            locality VARCHAR(45) NOT NULL,
                            locality_id INT NOT NULL REFERENCES localities(locality_id));
                         """)
        # pois table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS pois (
                            poi_id serial PRIMARY KEY,
                            poi_id_tourism VARCHAR(45) NOT NULL, 
                            label TEXT NOT NULL,
                            description TEXT,
                            last_update DATE NOT NULL, 
                            longitude NUMERIC NOT NULL,
                            latitude NUMERIC NOT NULL,
                            geom GEOMETRY(Point, 4326),
                            h3_5_id VARCHAR(45) NOT NULL REFERENCES h3_levels(h3_5_id),
                            address_id INT NOT NULL REFERENCES addresses(address_id)); 
                         
                         CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST(geom);
                         """)
        # poi_activity table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS poi_activity (
                            poi_id INT REFERENCES pois(poi_id),
                            activity_id INT REFERENCES activities(activity_id),
                            
                            PRIMARY KEY (poi_id, activity_id)
                         );
                        """)
        # poi_category table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS poi_category (
                            poi_id INT REFERENCES pois(poi_id),
                            category_id INT REFERENCES categories(category_id),
                            
                            PRIMARY KEY (poi_id, category_id)
                         );
                        """)
        
        # phone table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS phone_contact (
                            phone_id serial PRIMARY KEY,
                            phone_num VARCHAR(45) NOT NULL,
                            poi_id INT REFERENCES pois(poi_id));
                        """)
        
        # mail table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS mail_contact (
                            mail_id serial PRIMARY KEY,
                            mail_address VARCHAR(255) NOT NULL,
                            poi_id INT REFERENCES pois(poi_id));
                        """)
        
        # website table
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS website_contact (
                            website_id serial PRIMARY KEY,
                            website_address VARCHAR(2050) NOT NULL,
                            poi_id INT REFERENCES pois(poi_id));
                        """)
        self.conn.commit()
        print("creating tables in database : DONE ")
    
        
    def insert_into_tables(self, categories_df, poi_df, address_df, tel_df, email_df,  website_df, h3_level_df, locality_df) :
        # activity_table
        self.cur.execute("""
                         INSERT INTO activities (activity_duration)
                         VALUES (45), (60)
                        """)
        print('inserting values in activities table is done')

        categories = categories_df.iloc[:, 1:].drop_duplicates()

        # categories table 
        for i, row in categories.iterrows() :
            self.cur.execute("""
                             INSERT INTO categories (main_category, sub_category, itinerary_category) 
                             VALUES (%s, %s, %s)""",
                              (row['main_category'], row['sub_category'], row['to_keep'])
                              )
        print('inserting values in categories table is done')

        # localities table
        for i, row in locality_df.iterrows() :
            self.cur.execute("""
                             INSERT INTO localities (locality_name, locality_dep_name, locality_dep_code, locality_reg_name )
                             VALUES (%s, %s, %s, %s)""",
                             (row['locality'], row['dep_nom'], row['dep_code'], row['reg_nom'])
                             )
        print('inserting values in localities table is done')
            
        # h3 table :
        for i, row in h3_level_df.iterrows() :
            self.cur.execute("""
                             INSERT INTO h3_levels (h3_5_id, h3_7_id, h3_9_id )
                             VALUES (%s, %s, %s)""",
                             (row['h3_level_5'], row['h3_level_7'], row['h3_level_9'])
                             )
        print('inserting values in h3_levels table is done')
        
        # addresses table :
        for i, row in address_df.iterrows() :
            self.cur.execute( "SELECT locality_id FROM localities WHERE locality_name = %s ",
                             (row['locality'],)
                             )
            
            locality_ids = self.cur.fetchone()[0]

            self.cur.execute("""
                             INSERT INTO addresses (street_address, postal_code, locality, locality_id)
                             VALUES (%s, %s, %s, %s)""",
                             (row['street_adress'], int(row['postal_code']), row['locality'], locality_ids)
                             )
        print('inserting values in addresses table is done')
    
        # pois table  :
        for i, row in poi_df.iterrows() :
            poi_row = address_df.loc[address_df['dc:identifier'] == row['dc:identifier']] 
            poi_row = poi_row.iloc[0]        
            self.cur.execute("SELECT address_id FROM addresses WHERE (street_address = %s) AND (postal_code = %s) AND (locality = %s)",
                             (poi_row['street_adress'], int(poi_row['postal_code']), poi_row['locality'])
                             )
            
            address_ids = self.cur.fetchone()[0]

            self.cur.execute("""
                             INSERT INTO pois (poi_id_tourism, label, description, last_update, longitude, latitude, geom, h3_5_id, address_id)
                             VALUES (%s, %s, %s, %s, %s, %s, ST_MakePoint(%s, %s), %s, %s)""",
                             (row['dc:identifier'], row['label'], row['description'], row['lastUpdate'], row['longitude'], row['latitude'],
                              row['longitude'], row['latitude'], row['h3_level_5'] , address_ids)
                             )
        print('inserting values in pois table is done')

        # phone_contact table  :
        for i, row in tel_df.iterrows() :      
            self.cur.execute("SELECT poi_id FROM pois WHERE (poi_id_tourism = %s)",
                             (row['dc:identifier'], )
                             )
            
            poi_ids = self.cur.fetchone()[0]

            self.cur.execute("""
                             INSERT INTO phone_contact (phone_num, poi_id)
                             VALUES (%s, %s)""",
                             (row['Tel'], poi_ids)
                             )
        print('inserting values in phone_contact table is done')

        
        # mail_contact table  :
        for i, row in email_df.iterrows() :      
            self.cur.execute("SELECT poi_id FROM pois WHERE (poi_id_tourism = %s)",
                             (row['dc:identifier'], )
                             )
            
            poi_ids = self.cur.fetchone()[0]

            self.cur.execute("""
                             INSERT INTO mail_contact (mail_address, poi_id)
                             VALUES (%s, %s)""",
                             (row['email'], poi_ids)
                             )
        print('inserting values in mail_contact table is done')

        # mail_contact table  :
        for i, row in website_df.iterrows() :      
            self.cur.execute("SELECT poi_id FROM pois WHERE (poi_id_tourism = %s)",
                             (row['dc:identifier'], )
                             )
            
            poi_ids = self.cur.fetchone()[0]

            self.cur.execute("""
                             INSERT INTO website_contact (website_address, poi_id)
                             VALUES (%s, %s)""",
                             (row['Website'], poi_ids)
                             )
        print('inserting values in website_contact table is done')

        # poi_category table  :
        for i, row in categories_df.iterrows() :   

            self.cur.execute("SELECT poi_id FROM pois WHERE (poi_id_tourism = %s)",
                             (row['dc:identifier'], )
                             )
            
            poi_ids = self.cur.fetchone()[0]

            self.cur.execute("SELECT category_id FROM categories WHERE (main_category = %s) AND (sub_category = %s) AND (itinerary_category = %s)",
                             (row['main_category'], row['sub_category'], row['to_keep'])
                             )
            
            category_ids = self.cur.fetchone()[0]

            self.cur.execute("""
                             INSERT INTO poi_category (poi_id, category_id)
                             VALUES (%s, %s)
                             ON CONFLICT DO NOTHING""",
                             (poi_ids, category_ids)
                             )
        print('inserting values in poi_category table is done')
        self.conn.commit()
        print("inserting values in database : DONE ")