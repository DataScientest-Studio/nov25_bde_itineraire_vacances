"""
Module pour la transformation des donnnées extraites de DataTourisme et pour générer les dataframes pour chaque table de la base de données.
Les DataFrames brutes qui sont traitées sont :
    general_df : contient les informations générales du poi: nom, description, informations de contact(mail, adresse mail et adresse de site web)
    types_df : contient les types des pois l'objectif est de les mappers avec la catégorisation retenus
"""

import pandas as pd
import numpy as np
from pathlib import Path
from h3 import latlng_to_cell

from .utils import transform_utils as tu
from .config import config as cf



#------------------------------------------------------------------------------------------------
### Création de la table des catégories main_category/sub_category des POIs
#-------------------------------------------------------------------------------------------------

def transform_types_df(types_df) :
    categories_mapping_df = tu.get_categories_mapping_df()
    categories_df = types_df.merge(categories_mapping_df, how= 'left', left_on= 'type', right_on= 'linked_label')
    # filtrage des types de POI selon la valeur de 'to_keep'
    categories_df = categories_df[categories_df['to_keep'].isin([1, 2])] # filtrer sur 'to_keep' pour garder que les points de type 1 ou 2

    # sélection des colonnes nécessaires pour la construction de la table dans la BBD :
    categories_df = categories_df[['dc:identifier', 'main_category', 'sub_category', 'to_keep']]

    # Formattage des colonnes texte :
    text_col = ['dc:identifier', 'main_category', 'sub_category']
    categories_df.loc[:, text_col] = categories_df[text_col].astype(str)

    categories_df['to_keep'] = categories_df['to_keep'] == 1 #convertir les valeurs 1 en True et 2 en False pour l'inclusion du POI dans l'itinéraire ou pas.
    
    categories_df.drop_duplicates()

    return categories_df

#-------------------------------------------------------------------------------------------------------------------------
### Création des tables POI et de contacts (Adresse, siteweb, mail de contact, h3)  à partir de general_df et location df 
#-----------------------------------------------------------------------------------------------------------------------------

def transform_general_location_df(general_df, location_df) :
    """ retourne les tables poi/adress/phone_contact/mail/contact/website_contact/h3_level/locality à partir des dataframes general_df et location_df 
        issus du processus d'extraction.
    """
    
    description_df = general_df[['dc:identifier', 'label', 'description', 'lastUpdate']]
    email_df = general_df[['dc:identifier', 'email']]
    tel_df = general_df[['dc:identifier', 'Tel']]
    website_df = general_df[['dc:identifier', 'Website']]
    
    email_df, tel_df, website_df =[df.explode(df.columns[-1]) for df in [email_df, tel_df, website_df]]

    #===> nettoyage des données tel :

    # Application de la fonction de formatage aux numéros :
    tel_df['Tel'] = tel_df['Tel'].apply(tu.tel_format)
    # suppression des valeurs NaN et conversion des valeurs au format 'str'
    tel_df = tel_df.dropna().astype(str)
    
    #===> nettoyage des données website :
    website_df = website_df.dropna().astype(str)
   
    #===> nettoyage des données mails :
    email_df.dropna().astype(str)

    email_df['email']= email_df['email'].apply(tu.mail_format)
    email_df= email_df.dropna().astype(str)

   
    #===> nettoyage des colonnes de location_df :

    # formattage des colonnes non null:
    non_null_string_columns = ['dc:identifier', 'locality']
    location_df.loc[:, non_null_string_columns] = location_df[non_null_string_columns].astype(str)
    
    # formattage des colonnes numériques :
    numeric_columns = ['latitude', 'longitude']
    location_df.loc[:, numeric_columns] = location_df[numeric_columns].apply(pd.to_numeric, errors = 'coerce')

    location_df.loc[:, 'postal_code'] = (location_df['postal_code']
                                  .astype(str)
                                  .str.strip()
                                  .str.replace(' ', '')
                                  .replace('', None)
                                  .replace('nan', None)
                                  .astype('Int64')) 
    
    # formattage de la colonne street_adresse :
    location_df.loc[:, 'street_adress'] = location_df['street_adress'].apply(tu.adress_format)
    location_df.loc[:, 'street_adress'] = location_df['street_adress'].astype(str).replace('nan', 'unknown')

    #===> transformation de location_df en ajoutant les données des départements et de la région :
    ref_idf_df = tu.get_idf_localities_info()

    # merge de la table de référence de la région IDF avec location_df
    location_df = location_df.merge(ref_idf_df, how = 'left', left_on= 'locality', right_on= 'nom_standard')

    # traitement des localités qui sont des arrondissements de Paris
    mask_paris = location_df['locality'].str.contains('paris', case=False)
    ref_row = ref_idf_df[ref_idf_df['nom_standard'] == 'Paris'].iloc[0].values # racupération des valeurs assignées à Paris
    location_df.loc[mask_paris, location_df.columns[-11:]] = ref_row

    #===> création des table POI, Adresses et Communes :

    # création de la table locality :
    locality_col = ['locality', 'code_postal', 'codes_postaux', 'dep_code', 'dep_nom', 'reg_code', 'reg_nom', 
                    'latitude_mairie', 'longitude_mairie',
                    'latitude_centre', 'longitude_centre']
    locality_df = location_df[locality_col]
    locality_df = locality_df.drop_duplicates()

    # création de la table des adresses :
    address_df = location_df.iloc[: , :4]


    # création de la table des POI avec description et géolocalisation :
    poi_df= description_df.merge(location_df[['dc:identifier', 'latitude', 'longitude']],
                                 how = 'left',
                                 on = 'dc:identifier')
        ## Ajouts des catégorisations géographiques h3 des POI :
    poi_df['h3_level_5'] = poi_df.apply(lambda row: latlng_to_cell(row['latitude'], row['longitude'], 5), axis=1)
    poi_df['h3_level_7'] = poi_df.apply(lambda row: latlng_to_cell(row['latitude'], row['longitude'], 7), axis=1)
    poi_df['h3_level_9'] = poi_df.apply(lambda row: latlng_to_cell(row['latitude'], row['longitude'], 9), axis=1)

        ## création de la table des niveau h3: quartier, ville et département
    h3_level_df = poi_df[poi_df.columns[-3:]].drop_duplicates(subset= 'h3_level_5')

        ##création de la table des pois :
    poi_df = poi_df.drop(columns = poi_df.columns[-2:])
    poi_df = poi_df.drop_duplicates(subset= ['dc:identifier'])

    return poi_df, address_df, tel_df, email_df,  website_df, h3_level_df, locality_df    

if __name__ == "__main__" :
    print("ok")

