"""
Script d'extraction des données de POIs via un flux de données généré sur 
la plateforme Diffisueur DataToursime : https://diffuseur.datatourisme.fr/fr/
Le flux doit renvoyer les données sous format archive zip de fichiers JSON.
"""

from concurrent.futures import ThreadPoolExecutor
import requests
import io
from zipfile import ZipFile
import json
import pandas as pd

from .config import config as cf
from .utils import extract_utils as ext


class DataTourismeExtractor :

    def __init__(self, feed_url, app_key, timeout= 300, main_types = cf.main_types ):
        self.feed_url = feed_url # URL du webservice créé sur la plateforme DataTourisme Diffuseur
        self.app_key = app_key   # Clé de l'application créée sur la plateforme DataTourisme Diffuseur
        self.url = feed_url + '/' + app_key
        self.timeout = timeout
        self.main_types = main_types

    # récupération des données sous forme d'un dataframe:
    def get_raw_poi_df(self):
        try:
            response = requests.get(self.url, timeout=self.timeout)
            response.raise_for_status()

            with ZipFile(io.BytesIO(response.content)) as zipfile:
                # Récupération de l'index
                with zipfile.open("index.json") as f:
                    try:
                        index_df = pd.read_json(f)
                    except Exception as e:
                        print(f"Erreur à la lecture de l'index : {e}")

                # Fonction pour charger un fichier JSON
                def load_json_file(file):
                    with zipfile.open('objects/' + file) as f:
                        data = json.load(f)
                    return pd.json_normalize(data, errors='ignore', sep='_')

                # Lecture parallèle des fichiers avec 6 workers :
                df_list = []
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = [executor.submit(load_json_file, file) for file in index_df['file']]
                    for future in futures:
                        df_list.append(future.result())

                # Concaténation et traitement
                poi_df = pd.concat(df_list, axis=0, ignore_index=False)
                poi_df = ext.vectorized_simple_list_extract(poi_df)

                # Ajout de l'index
                poi_df.index = index_df.index
                poi_df = pd.concat([index_df, poi_df], axis=1)

                # Sélection des colonnes
                columns = ['dc:identifier', 'label', '@type', 'hasTheme', 'hasContact', 'hasDescription', 'rdfs:comment_fr',
                        'isLocatedAt', 'hasReview', 'offers',
                        'lastUpdate', 'lastUpdateDatatourisme', 'schema:startDate', 'schema:endDate',
                        'takesPlaceAt', 'providesCuisineOfType', 'reducedMobilityAccess']
                poi_df = poi_df[columns]

                print(f"{poi_df.shape[0]} POI ont été importés")

                self.poi_df = poi_df
                return self.poi_df

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête : {e}")

    
    def download_poi_zip(self, path = "data_file.zip"):
        """ récupération des données et création d'un zip sur le disque dur """
        try :
            response = requests.get(self.url, timeout=self.timeout)
            response.raise_for_status()

            
            with open(path, 'wb') as f :
                f.write(response.content)
            
            print(f"ZIP téléchargé ({len(response.content) / 1024 / 1024:.2f} MB)\n")

        except requests.exceptions.RequestException as e :
            print(f"Erreur lors de la requête : {e}")
            return None

    def get_general_df(self) :
            if hasattr(self, "poi_df"): # vérifie sur l'attribut poi_df existe :
                # extraction des données de description des POI
                description_df = pd.json_normalize(data = self.poi_df['hasDescription']).map(ext.simple_list_extract)
                description_df = description_df[['shortDescription.fr']].rename(columns= {'shortDescription.fr' : 'description'})
                description_df.index = self.poi_df[['dc:identifier']].index
                
                # extraction des données de contact des POI
                contact_df = pd.json_normalize(data = self.poi_df['hasContact']).map(ext.simple_list_extract)
                columns_to_keep = ['schema:email','schema:telephone','foaf:homepage']
                new_columns_dict ={'schema:email' : 'email', 'schema:telephone' : 'Tel', 'foaf:homepage' : 'Website'}
                contact_df = contact_df[columns_to_keep].rename(columns= new_columns_dict)
                contact_df.index = self.poi_df[['dc:identifier']].index

                general_df =  pd.concat([self.poi_df[['dc:identifier', 'lastUpdate', 'label']],description_df, contact_df ], axis= 1)
                
                self.general_df = general_df

                return self.general_df
            
            else :
                print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")

    def get_types_df(self) :
        if hasattr(self, 'poi_df') :
            types_df = self.poi_df[['dc:identifier','@type']]
            types_df = types_df.explode('@type')
            types_df['@type'] = types_df['@type'].str.replace("schema:", '')
            types_df = types_df.rename(columns  = {'@type' : 'type'})
            types_df = types_df.drop_duplicates()
            types_df = types_df[types_df['type'] != 'PointOfInterest']

            self.types_df = types_df
            
            return self.types_df
        else : 
            print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")


    def get_themes_df(self):
        """ extraction des thèmes et sous_thèmes des POI """

        if hasattr(self, 'poi_df') :
            themes_df = ext.poi_structure_extract(self.poi_df, 'hasTheme')

            themes_df = themes_df[['dc:identifier', '@type', 'rdfs:label.fr']] # récupération d'une partie des colonnes
            themes_df = themes_df.rename(columns = {'@type' : 'theme', 'rdfs:label.fr' : 'sub_theme'}) # renomage des colonnes
            themes_df = themes_df.explode('theme') # répartir les valeurs de thèmes sous forme de liste sur plusieurs lignes

            self.themes_df = themes_df
            return self.themes_df
        else : 
            print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")
    
    def get_location_df(self) :
        """extracttion des données de localisation des POI"""
        if hasattr(self, 'poi_df'):

            location_df = pd.json_normalize(data = self.poi_df['isLocatedAt'], max_level = 1 ,
                                        meta = [["schema:geo","schema:latitude"],
                                                ["schema:geo","schema:longitude"]], 
                                            meta_prefix = '_',
                                            record_path= ['schema:address'],
                                            record_prefix = 'adress_',
                                            errors= 'ignore', sep = '_')

            columns_to_keep = ['adress_schema:addressLocality',
                            'adress_schema:postalCode',
                            'adress_schema:streetAddress',
                            '_schema:geo_schema:latitude',
                            '_schema:geo_schema:longitude']

            new_columns_dict = {'adress_schema:addressLocality' : 'locality' ,
                                'adress_schema:postalCode' :'postal_code',
                                'adress_schema:streetAddress' : 'street_adress',
                                '_schema:geo_schema:latitude' : 'latitude',
                                '_schema:geo_schema:longitude' : 'longitude'}

            location_df = location_df[columns_to_keep].rename(columns = new_columns_dict) # choix des colonnes et renommage

            # transformation des valeurs sous forme de liste d'un seul élément :
            location_df = location_df.map(ext.simple_list_extract)

            # concaténation avec les id des POI
            location_df.index = self.poi_df[['dc:identifier']].index
            location_df = pd.concat([self.poi_df[['dc:identifier']], location_df], axis= 1)

            self.location_df = location_df

            return self.location_df
        else :
            print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")
        
    def get_opening_hours_df(self) :
        """extracttion des données des horaires d'ouverture des POI"""
        
        if hasattr(self, 'poi_df'):

            is_located_at_df = pd.json_normalize(data = self.poi_df['isLocatedAt'])

            is_located_at_df = pd.concat([self.poi_df['dc:identifier'], is_located_at_df], axis=1)

            # extraction des données : Id du POI et la structure 'schema: openingHoursSpecification'
            # application de la fonction explode() pour avoir un dictionnaire 'schema: openingHoursSpecification' par ligne.
            opening_hours_df = is_located_at_df[['dc:identifier','schema:openingHoursSpecification']].explode('schema:openingHoursSpecification')

            # répartition des données du dictionnaire 'schema: openingHoursSpecification' sur plusieurs colonnes 
            schema_opening_hours_df= pd.json_normalize(data = opening_hours_df['schema:openingHoursSpecification'])

            # suppression des colonnes de traduction des infos supplémentaires :
            schema_opening_hours_df = schema_opening_hours_df.drop(columns = ['@type', 'hasTranslatedProperty', 'additionalInformation.de', 'additionalInformation.en',
                                                            'additionalInformation.it', 'additionalInformation.nl',	'additionalInformation.es'])
            # application de la fonction simple_list_extract pour transformer le contenu de la colonne additionalInformation.fr de list à string
            schema_opening_hours_df = schema_opening_hours_df.map(ext.simple_list_extract)

            # concaténation des deux df en s'assurant qu'elles ont le même index:
            schema_opening_hours_df.index =  opening_hours_df[['dc:identifier']].index                       
            opening_hours_df= pd.concat([opening_hours_df[['dc:identifier']], schema_opening_hours_df], axis= 1)

            opening_hours_df = opening_hours_df.reset_index(drop=True)

            self.opening_hours_df =  opening_hours_df

            return self.opening_hours_df
        else :
            print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")

    def get_reviews_df(self) :
        """extracttion des données de classement des POI"""
        
        if hasattr(self, 'poi_df'):        
            reviews_df = ext.poi_structure_extract(self.poi_df, 'hasReview')

            # sélection d'un partie des données :
            reviews_df = reviews_df[['dc:identifier', 'hasReviewValue.@type', 'hasReviewValue.rdfs:label.fr', 
                                'hasReviewValue.isCompliantWith', 'hasReviewValue.schema:ratingValue' ]]
            # renommer les colonnes :
            rename_col_dict = {'dc:identifier' : 'poi_id',  
                        'hasReviewValue.@type' : 'review_category', 
                        'hasReviewValue.rdfs:label.fr' : 'review_value', 
                        'hasReviewValue.isCompliantWith' : 'compliant_with' , 
                        'hasReviewValue.schema:ratingValue' : 'rating_value'}
            reviews_df = reviews_df.rename(columns = rename_col_dict)
            
            self.reviews_df = reviews_df

            return self.reviews_df
        
        else :
            print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")

    def get_offers_df(self):
        """extracttion des données de classement des POI"""
        
        if hasattr(self, 'poi_df'):

            # extraction de la structure 'shema:priceSpecification' sous forme de df
            offers = pd.json_normalize(data = self.poi_df['offers'])
            offers.index = self.poi_df[['dc:identifier']].index

            offers_df_raw = pd.concat([self.poi_df[['dc:identifier']], offers], axis = 1)
            offers_df_raw = offers_df_raw[['dc:identifier', 'schema:priceSpecification']]

            # extraction des données de la structure 'shema:priceSpecification' sur plusieurs colonnes :
            df = ext.poi_structure_extract(offers_df_raw, 'schema:priceSpecification')

            # séparation des données de la structure 'hasPricingOffer'
            offers = pd.json_normalize(data = df['hasPricingOffer']).map(ext.simple_list_extract)
            offers = offers[['@type', 'rdfs:label.fr']].rename(columns = {'@type' : 'pricing_offer_id', 'rdfs:label.fr' : 'pricing_offer_label'})
            offers.index = df.index

            # séparation des données de la structure 'appliesOnPeriod'
            periods = pd.json_normalize(data = df['appliesOnPeriod'])
            periods = periods[['startDate', 'endDate']].rename(columns = {'startDate' : 'pricing_start_date', 'endDate' : 'pricing_end_date'})
            periods.index = df.index

            # séparation des données de la structure 'hasEligiblePolicy'
            policies = pd.json_normalize(data = df['hasEligiblePolicy'])
            policies = policies[['@id', 'rdfs:label.fr']].rename(columns = {'@id' : 'pricing_policy_id', 'rdfs:label.fr' : 'pricing_policy_label'})
            policies['pricing_policy_id'] = policies['pricing_policy_id'].str.replace('kb:', '')
            policies = policies.map(ext.simple_list_extract)
            policies.index = df.index

            # séparation des données de la structure 'hasPricingMode'
            modes = pd.json_normalize(data = df['hasPricingMode'])
            modes = modes[['@id', 'rdfs:label.fr']].rename(columns = {'@id' : 'pricing_mode_id', 'rdfs:label.fr' : 'pricing_mode_label'})
            modes.index = df.index

            # sélection des colonnes à garder  :
            columns_to_keep = ['dc:identifier', 'schema:minPrice', 'schema:maxPrice', 'schema:priceCurrency', 'name.fr']
            columns_new_names = {'schema:minPrice': 'min_price', 
                                'schema:maxPrice' : 'max_price', 
                                'schema:priceCurrency' : 'currency_price',
                                'name.fr' : 'offer_description'}

            # concaténation de l'ensemble des dataframes en une seule avec sélection et renomage des colonnes :
            offers_df = pd.concat([df[columns_to_keep], periods, offers, policies, modes] , axis= 1)
            offers_df = offers_df.rename(columns = columns_new_names)
            
            self.offers_df = offers_df

            return self.offers_df
        else :
            print("Vous devez créer l'attribut 'poi_df' avec la méthode 'get_raw_poi_df' pour pouvoir utiliser cette méthode ")


if __name__ == "__main__" :
    import cProfile
    import pstats
    feed_url = 'https://diffuseur.datatourisme.fr/webservice/926c193f11cc9605128f64a6484bfceb' #url du web service pour POI IDF
    app_key =  '69975793-4d7b-4a8f-b3ae-41cae27b846c' 
    extractor =  DataTourismeExtractor(feed_url, app_key)

    # Création et activation du Profiler :
    profiler = cProfile.Profile()
    profiler.enable()

    # extraction des données brutes des POI :
    extractor.get_raw_poi_df()
   
    # extraction des types des POI
    types_df =extractor.get_types_df()

    level_1_types_df = types_df[types_df['type'].isin(cf.main_types_linked)]
    print("\n répartition des POI par type en % :\n")
    print(level_1_types_df.groupby('type')['dc:identifier'].count().apply(lambda x : round( x/ level_1_types_df.shape[0] *100, 2)))

    # désactivation du profiler :
    profiler.disable()

    print("statistique d'exécution du script :")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
