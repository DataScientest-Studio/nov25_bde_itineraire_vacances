"""
Pipeline ETL pour extraire, transformer et charger les points d'intérêt (POI) de la région Île-de-France(IDF) depuis le flux Datatourisme vers une base de données PostgreSQL/PostGIS
Il est possible de remplacer le flux de données POI de la région IDF par un flux similaire (zip de fichiers JSON) sur un périmètre géographique différent.

"""
from datetime import datetime
import cProfile
import pstats

from etl import extract_opt as extract
from etl import transform
from etl import load


# Création et activation du Profiler :
profiler = cProfile.Profile()
profiler.enable()

#-----------------------------------------------------------------------------
#### extraction de la donnée
#-----------------------------------------------------------------------------
now = datetime.now()
print(f"{now.strftime("%H:%M:%S")}: lancement de l'extraction")


# Récupération des POI de la région IdF directement depuis le flux Datatourisme :
feed_url = 'https://diffuseur.datatourisme.fr/webservice/926c193f11cc9605128f64a6484bfceb' # URL du webservice
app_key = '69975793-4d7b-4a8f-b3ae-41cae27b846c' # clé de l'application

# initialisation de l'extracteur avec les données du flux :
extractor = extract.DataTourismeExtractor(feed_url= feed_url, 
                                          app_key= app_key)

# extraction des données brutes :
extractor.get_raw_poi_df()
now = datetime.now()
print(f"{now.strftime("%H:%M:%S")}: extraction du dataframe de données brutes des pois terminée")

# extraction des données générales des POI :
general_df = extractor.get_general_df()
# extraction des types des POI
types_df =extractor.get_types_df()
# extraction des données de géolocalisation
location_df = extractor.get_location_df()
now = datetime.now()
print(f"{now.strftime("%H:%M:%S")}: extraction des 3 dataframes general_df, location_df, types_df terminée, lancement de la transformation")

#-----------------------------------------------------------------------------
#### Transformation
#-----------------------------------------------------------------------------

# catégorisation des pois : 
categories_df = transform.transform_types_df(types_df)

# filtrage pour ne garder que les pois avec une catégorisation :
poi_index = categories_df['dc:identifier'].unique()
general_df = general_df[general_df['dc:identifier'].isin(poi_index)]
location_df = location_df[location_df['dc:identifier'].isin(poi_index)]

# création et transformation des tables à partir de la table de description (general_df) et la table de localisation (location_df) :
poi_df, address_df, tel_df, email_df,  website_df, h3_level_df, locality_df = transform.transform_general_location_df(general_df, location_df)
now = datetime.now()
print(f"{now.strftime("%H:%M:%S")}: transformation des données terminée, lancement du chargement dans la base de données")

#-----------------------------------------------------------------------------
#### Loading
#-----------------------------------------------------------------------------

dbm =load.DBManager(db_env= 'postgres_postgis/.env')
dbm.create_tables()
dbm.insert_into_tables(categories_df, poi_df, address_df, tel_df, email_df,  website_df, h3_level_df, locality_df)

now = datetime.now()
print(f"{now.strftime("%H:%M:%S")}: Les données sont disponibles dans la base de donnée :)")

#-----------------------------------------------------------------------------
 # désactivation du profiler et affichage des statistiques :
profiler.disable()

print("statistique d'exécution du script :")
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)