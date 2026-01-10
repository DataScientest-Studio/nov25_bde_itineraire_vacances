
import sys
from pathlib import Path
import pandas as pd
import requests
import re
import numpy as np
from . import extract_utils as eu


# Ajouter le dossier parent au chemin
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import main_types_fr, new_categories_mapping_df, main_types


class ClassExtract() :
    def __init__(self, 
                 path = 'classes_en.csv', 
                 url = 'https://gitlab.adullact.net/adntourisme/datatourisme/ontology/-/raw/master/Documentation/classes_en.csv') :
        self.url = url
        self.path = path
    
    def download_class_file(self, path = 'classes_en.csv') :

        """ téléchargement du fichier csv"""
        try :
            response = requests.get(self.url)
            response.raise_for_status()
            
            with open(path, 'wb') as f :
                f.write(response.content)
            
            self.path = path
            print(f"Fichier téléchargé !\n")

        except requests.exceptions.RequestException as e :
            print(f"Erreur lors de la requête : {e}")
            return None

    def get_main_types_df(self, main_types = main_types ):
        """ récupération des types de POI sur 4 niveaux """
        # extraction des données du fichier csv :

        classes_df = pd.read_csv(self.path, skiprows= 1, names= ['linked_label', 'label', 'linked_parent_label', 'parent_label'] )
        classes_df['linked_label'] = classes_df['linked_label'].str.replace('<https://www.datatourisme.fr/ontology/core#', '')
        classes_df['linked_parent_label'] = classes_df['linked_parent_label'].str.replace('<https://www.datatourisme.fr/ontology/core#', '')

        classes_df['linked_label'] = classes_df['linked_label'].str.replace('>', '')
        classes_df['linked_parent_label'] = classes_df['linked_parent_label'].str.replace('>', '')

        classes_df= classes_df[['parent_label', 'linked_parent_label', 'label' ,'linked_label']]

        # filtrage des donénes pour ne garder que les types de POI :
        main_types_df = classes_df[classes_df['parent_label'].isin(main_types)]

        # récupération du niveau 3 des types :
        main_types_df = main_types_df.merge(classes_df, how= 'left', left_on= 'label', right_on= 'parent_label')
        main_types_df = main_types_df.drop(columns=['parent_label_y',	'linked_parent_label_y'])
        main_types_df = main_types_df.rename(columns= {'parent_label_x' : 'level_1_label', 
                                                            'linked_parent_label_x' : 'level_1_linked_label',
                                                            'label_x' : 'level_2_label',
                                                            'linked_label_x' : 'level_2_linked_label',
                                                            'label_y' : 'level_3_label',
                                                            'linked_label_y' : 'level_3_linked_label',
                                                            })
        # récupération du niveau 3 des types :
        main_types_df = main_types_df.merge(classes_df, how= 'left', left_on= 'level_3_label', right_on= 'parent_label')
        main_types_df = main_types_df.drop(columns=['parent_label',	'linked_parent_label'])
        main_types_df = main_types_df.rename(columns= {'label' : 'level_4_label', 
                                                            'linked_label' : 'level_4_linked_label'})
        return main_types_df


ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT /"config"
input_classes_fr_path = INPUT_DIR / 'classes_fr.csv'
input_categories_mapping_path = INPUT_DIR / 'poi_category_mapping_251231.csv'
input_localities_path = INPUT_DIR / 'communes-france-2025.csv'



def get_categories_mapping_df(classes_path= input_classes_fr_path , categories_mapping_path= input_categories_mapping_path, main_types = main_types_fr) :
    """" 
    Création d'un dataframe simplifiée de mapping entre les types de DataTourisme et les catagories définies pour le projet.
    args : 
        classes_path : Chemin vers la matrices des classes de Datatourisme
        categories_mapping_path : tableau de mapping défini dans le cadre du projet
        main_types = liste des 4 valeurs des types de POI sur DataTourisme dans la langue du fichier des classes

    """
    extracor =  ClassExtract(path= input_classes_fr_path)
    main_types_df = extracor.get_main_types_df(main_types )

    # extraction du tableau de mapping :
    categories_mapping_df = pd.read_csv(input_categories_mapping_path).drop(columns = 'index')


    ### fusion de la matrice des types avec celle du mapping pour récupérer le linked_label associé au label_3_4, qui vont nous permettre de filtrer les POI et leur associer
    ## les nouvelles catégories :

    df= pd.merge(main_types_df, categories_mapping_df,
                 how = 'left', 
                 on= ['level_1_label', 'level_2_label', 'level_3_label', 'level_4_label'])

    # ajout du linked_label associée à la colonne 'Level 3/4' :
    df.loc[df['Level_3_4'] == df['level_2_label'], 'linked_label'] = df.loc[df['Level_3_4'] == df['level_2_label'], 'level_2_linked_label']
    df.loc[df['Level_3_4'] == df['level_3_label'], 'linked_label'] = df.loc[df['Level_3_4'] == df['level_3_label'], 'level_3_linked_label']
    df.loc[df['Level_3_4'] == df['level_4_label'], 'linked_label'] = df.loc[df['Level_3_4'] == df['level_4_label'], 'level_4_linked_label']

    # ajout des nouvelles catégories issus de l'analyse des POIs
    df = pd.concat([new_categories_mapping_df,
                    df[['to_keep', 'main_category', 'sub_category', 'linked_label']]], 
                    axis= 0)

    return df

def get_poi_index(df, id = 'dc:identifier') :
    """
    retourne l'index de POI unique d'un dataframe avec la colonne 'dc:identifier'
    
    """
    return df[id].unique()

def tel_format(phone):
    # définition des patterns regex pour filtrer les numéros de téléphone valides :
    tel_pattern_1 = r'^\+(?:33|590|596|262|594|269|508|689|687|681) \d \d{2} \d{2} \d{2} \d{2}$' # prises en comptes des indicatifs des département outre-mer
    tel_pattern_2 = r'^\+(?:33|590|596|262|594|269|508|689|687|681) \d{4}$' # numéro public de type +33 XXXX
    if (pd.isna(phone)) or (phone is None) :
        return np.nan
    elif re.match(tel_pattern_1, phone) or re.match(tel_pattern_2, phone) :
        return phone
    else :            
        if len(phone) == 16 :
            # Supprimer tous les espaces
            phone = phone.replace(' ', '')
            # Ajouter les espaces au bon endroit
            return f"{phone[:3]} {phone[3]} {phone[4:6]} {phone[6:8]} {phone[8:]}"
        elif len(phone) == 17 :
            # Supprimer tous les espaces
            phone = phone.replace(' ', '')
            # Ajouter les espaces au bon endroit
            return f"{phone[:4]} {phone[4]} {phone[5:7]} {phone[7:9]} {phone[9:]}"
        else :
            return np.nan

def mail_format(mail):
    #définition d'une fonction de validation minimale d'un mail :
    mail_pattern = r'^.+@.+\..+$'
    if (pd.isna(mail)) or (mail is None) :
        return np.nan
    
    elif re.match(mail_pattern, mail) :
        return mail
    else :
        return np.nan

def adress_format(adress) : 
    # formattage de street_adresse qui peut être une liste :
    if isinstance(adress, list) :
        if len(adress) == 0 :
            return np.nan
        elif len(adress) == 1 :
            return eu.simple_list_extract(adress)
        else :
            new_adress = ' '.join(str(element) for element in adress)
            return new_adress
    else :
        return adress

def get_idf_localities_info():
    # Récupération d'un fichier des communes avec association au département et région :
    columns = ['nom_standard', 'code_postal', 'codes_postaux', 'reg_code', 'reg_nom', 'dep_code', 'dep_nom',
           'latitude_mairie', 'longitude_mairie', 'latitude_centre', 'longitude_centre']

    ref_df = pd.read_csv(input_localities_path, usecols= columns )

    # filtrage sur la région Île-de-France
    ref_idf_df = ref_df[ref_df['reg_nom'] == 'Île-de-France']

    return ref_idf_df

if __name__ == "__main__" :
    
    df = get_categories_mapping_df(classes_path= input_classes_fr_path , categories_mapping_path= input_categories_mapping_path, main_types = main_types_fr)
    
    df.to_csv(INPUT_DIR/'last_mapping_matrix.csv')
    print(f"la nouvelle matrice du mapping de catégories a été générée et enregistrée dans {INPUT_DIR/'last_mapping_matrix.csv'}, elle est contient {df.shape[0]} lignes")