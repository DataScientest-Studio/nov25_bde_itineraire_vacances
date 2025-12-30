import requests
import io
from zipfile import ZipFile
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


main_types = ['Place', 'Route', 'Product', 'Entertainment and event']

# récupération des données sous forme de dataframe pandas:
def poi_df(feed_url):
    try :
        response = requests.get(feed_url)
        response.raise_for_status()

        # Création d'un objet binaire avec le contenu de la réponse dans la RAM
        ram_object = io.BytesIO(response.content)

        # accès au zipfile avec le contenu du flux
        with ZipFile(ram_object) as zipfile : 
            # récupération de l'index des objects json
            with zipfile.open("index.json") as f :
                try :
                    index_df = pd.read_json(f)
                except Exception as e :
                    print(f"Erreur à la lecture de l'index : {e}")
        
            # création d'un dataframe avec les données des POI
            poi_df= pd.DataFrame()

            for file in index_df['file'] :
                with zipfile.open('objects/'+ file) as f :
                    data = json.load(f)
                f_df = pd.json_normalize(data, 
                                    errors= 'ignore', 
                                    sep = '_')
    
                poi_df = pd.concat([poi_df, f_df], axis = 0)
        
        # ajout de l'index pour récupérer le nom et la date de la dernière mise à jour du POI :
        poi_df.index = index_df.index
        poi_df = pd.concat([index_df, poi_df], axis = 1)
        
        print(f"{poi_df.shape[0]} POI ont été importés")
        return poi_df
    
    except requests.exceptions.RequestException as e :
        print(f"Erreur lors de la requête : {e}")
        return None

# récupération des données et création d'un zip sur le disque dur :
def get_all_data(feed_url, path):
    try :
        response = requests.get(feed_url, timeout= 10)
        response.raise_for_status()

        
        with open(path, 'wb') as f :
            f.write(response.content)
        
        print(f"ZIP téléchargé ({len(response.content) / 1024 / 1024:.2f} MB)\n")

    except requests.exceptions.RequestException as e :
        print(f"Erreur lors de la requête : {e}")
        return None
    

# définition d'une fonction pour transformer une valeur de dataframe d'une liste d'un élément à un élement :
def simple_list_extract(x) :
    if (isinstance(x,list)) and (len(x) == 1) :
       return x[0]
    else :
        return x

# definition fct pour plotter le % de valeurs manquantes par colonne pour df de  :
def poi_na_plot(df) :
    data = df.isna().sum().map(lambda x : round(x/df.shape[0]*100)).sort_values()
    ax = sns.barplot(x = data.values, y = data.index)
    ax.bar_label(ax.containers[0], fontsize = 10)
    ax.set_xlim(0, 100)
    plt.title("Pourcentage de POI SANS la structure de données indiquée")
    plt.ylabel("Structure de données")
    plt.xlabel("Pourcentage de POI (%)")

def poi_structure_extract(poi_df, data_struct):
    df_raw = poi_df[['dc:identifier', data_struct]]
    
    #création de 4 masks pour diviser le dataframe en 4 sections : 

    # sans thème (valeur null) :
    mask_null = df_raw[data_struct].isna()

    # avec theme sous forme de dictionnaire :
    mask_dict = df_raw[data_struct].apply(lambda x : (isinstance(x,dict)))

    # avec une structure hastheme sous forme d'une liste d'un seul élément 
    mask_simple_list = df_raw[data_struct].apply(lambda x : (isinstance(x,list)) and (len(x)==1) )

    ## avec une structure hastheme sous forme d'une liste d'un seul élément 
    mask_multiple_list = df_raw[data_struct].apply(lambda x : (isinstance(x,list)) and (len(x) > 1) )

    # répartition des éléments de la liste sur plusieurs lignes, une ligne par élément :
    if mask_multiple_list.sum() > 0 :
        simple_list_df = df_raw[mask_multiple_list].explode(data_struct)
    else :
        simple_list_df = pd.DataFrame()
    
    # concaténation avec les poi ayant une structure 'hasTheme' simple :
    simple_list_df = pd.concat([simple_list_df,
                                df_raw[mask_simple_list].map(simple_list_extract),
                                df_raw[mask_dict] ], axis = 0, ignore_index= True)

    if len(simple_list_df) > 0 :

        # répartition du dictionnaire 'hasTheme' sur plusieurs colonnes, une colonne par clé :
        columns_df = pd.json_normalize(data = simple_list_df[data_struct])

        # ajout des identifiants des POI
        columns_df.index = simple_list_df.index
        df = pd.concat([simple_list_df[['dc:identifier']], columns_df], axis = 1)

        if mask_null.sum() > 0 :
        # ajout des POI sans structure 'hasTheme'
            df = pd.concat([df, df_raw[mask_null]], axis = 0, ignore_index= True).map(simple_list_extract)

    else :
        df = simple_list_df

    return df

def poi_general_extract(poi_df) :

    description_df = pd.json_normalize(data = poi_df['hasDescription']).map(simple_list_extract)
    description_df = description_df[['shortDescription.fr']].rename(columns= {'shortDescription.fr' : 'description'})
    description_df.index = poi_df[['dc:identifier']].index

    contact_df = pd.json_normalize(data = poi_df['hasContact']).map(simple_list_extract)
    columns_to_keep = ['schema:email','schema:telephone','foaf:homepage']
    new_columns_dict ={'schema:email' : 'email', 'schema:telephone' : 'Tel', 'foaf:homepage' : 'Website'}
    contact_df = contact_df[columns_to_keep].rename(columns= new_columns_dict)
    contact_df.index = poi_df[['dc:identifier']].index

    df =  pd.concat([poi_df[['dc:identifier', 'label','rdfs:comment_fr']],description_df, contact_df ], axis= 1)
    df = df.rename(columns= {'rdfs:comment_fr' : 'comment'})

    return df

def poi_types_extract(poi_df) :

    df= poi_df[['dc:identifier','@type']]
    df = df.explode('@type')
    df['@type'] = df['@type'].str.replace("schema:", '')
    df = df.rename(columns  = {'@type' : 'type'})
    df = df.drop_duplicates()
    return df

def poi_themes_extract(poi_df):
    themes_df = poi_structure_extract(poi_df, 'hasTheme')

    themes_df = themes_df[['dc:identifier', '@type', 'rdfs:label.fr']] # récupération de certaines colonnes
    themes_df = themes_df.rename(columns = {'@type' : 'theme', 'rdfs:label.fr' : 'sub_theme'}) # renomage des colonnes
 
    return themes_df

def poi_location_extract(poi_df) :
    # traitement de la description du POI :


    location_df = pd.json_normalize(data = poi_df['isLocatedAt'], max_level = 1 ,
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

    location_df = location_df[columns_to_keep].rename(columns = new_columns_dict)

    # Modifier le format des valeurs sous forme de list d'un seul élément :
    location_df = location_df.map(simple_list_extract)

    # concaténation avec les id des POI
    location_df.index = poi_df[['dc:identifier']].index
    location_df = pd.concat([poi_df[['dc:identifier']], location_df], axis= 1)

    return location_df

def poi_opening_hours_extract(poi_df) :
    is_located_at_df = pd.json_normalize(data = poi_df['isLocatedAt'])

    is_located_at_df = pd.concat([poi_df['dc:identifier'], is_located_at_df], axis=1)

    # extraction des données : Id du POI et la structure 'schema: openingHoursSpecification'
    # application de la fonction explode() pour avoir un dictionnaire 'schema: openingHoursSpecification' par ligne.
    opening_hours_df = is_located_at_df[['dc:identifier','schema:openingHoursSpecification']].explode('schema:openingHoursSpecification')

    # répartition des données du dictionnaire 'schema: openingHoursSpecification' sur plusieurs colonnes 
    schema_opening_hours_df= pd.json_normalize(data = opening_hours_df['schema:openingHoursSpecification'])

    # suppression des colonnes de traduction des infos supplémentaires :
    schema_opening_hours_df = schema_opening_hours_df.drop(columns = ['@type', 'hasTranslatedProperty', 'additionalInformation.de', 'additionalInformation.en',
                                                    'additionalInformation.it', 'additionalInformation.nl',	'additionalInformation.es'])
    # application de la fonction simple_list_extract pour transformer le contenu de la colonne additionalInformation.fr de list à string
    schema_opening_hours_df = schema_opening_hours_df.map(simple_list_extract)

    # concaténation des deux df en s'assurant qu'elles sont le même POI :
    schema_opening_hours_df.index =  opening_hours_df[['dc:identifier']].index                       
    opening_hours_df= pd.concat([opening_hours_df[['dc:identifier']], schema_opening_hours_df], axis= 1)

    opening_hours_df = opening_hours_df.reset_index(drop=True)

    return opening_hours_df

def poi_review_extract(poi_df) :
    reviews_df = poi_structure_extract(poi_df, 'hasReview')

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
    
    return reviews_df

def poi_offers_extract(poi_df):
    # extraction de la structure 'shema:priceSpecification' sous forme de df
    offers = pd.json_normalize(data = poi_df['offers'])
    offers.index = poi_df[['dc:identifier']].index

    offers_df_raw = pd.concat([poi_df[['dc:identifier']], offers], axis = 1)
    offers_df_raw = offers_df_raw[['dc:identifier', 'schema:priceSpecification']]

    # extraction des données de la structure 'shema:priceSpecification' sur plusieurs colonnes :
    df = poi_structure_extract(offers_df_raw, 'schema:priceSpecification')

    # séparation des données de la structure 'hasPricingOffer'
    offers = pd.json_normalize(data = df['hasPricingOffer']).map(simple_list_extract)
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
    policies = policies.map(simple_list_extract)
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

    # concaténation de l'ensemble des dataframes en une seule 
    offers = pd.concat([df[columns_to_keep], periods, offers, policies, modes] , axis= 1)
    offers = offers.rename(columns = columns_new_names)
    
    return offers

def classes_extract (path = 'classes_en.csv'):
# récupération des catégories des POI :
    classes_df = pd.read_csv(path, skiprows= 1, names= ['linked_label', 'label', 'linked_parent_label', 'parent_label'] )
    classes_df['linked_label'] = classes_df['linked_label'].str.replace('<https://www.datatourisme.fr/ontology/core#', '')
    classes_df['linked_parent_label'] = classes_df['linked_parent_label'].str.replace('<https://www.datatourisme.fr/ontology/core#', '')

    classes_df['linked_label'] = classes_df['linked_label'].str.replace('>', '')
    classes_df['linked_parent_label'] = classes_df['linked_parent_label'].str.replace('>', '')

    classes_df= classes_df[['parent_label', 'linked_parent_label', 'label' ,'linked_label']]
    return classes_df

def types_tree(classes_df) :
    main_types_df = classes_df[classes_df['parent_label'].isin(main_types)]

    main_types_df = main_types_df.merge(classes_df, how= 'left', left_on= 'label', right_on= 'parent_label')
    main_types_df = main_types_df.drop(columns=['parent_label_y',	'linked_parent_label_y'])
    main_types_df = main_types_df.rename(columns= {'parent_label_x' : 'level_1_label', 
                                                         'linked_parent_label_x' : 'level_1_linked_label',
                                                         'label_x' : 'level_2_label',
                                                         'linked_label_x' : 'level_2_linked_label',
                                                         'label_y' : 'level_3_label',
                                                         'linked_label_y' : 'level_3_linked_label',
                                                         })

    main_types_df = main_types_df.merge(classes_df, how= 'left', left_on= 'level_3_label', right_on= 'parent_label')
    main_types_df = main_types_df.drop(columns=['parent_label',	'linked_parent_label'])
    main_types_df = main_types_df.rename(columns= {'label' : 'level_4_label', 
                                                         'linked_label' : 'level_4_linked_label'})
    return main_types_df