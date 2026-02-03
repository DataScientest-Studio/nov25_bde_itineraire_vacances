import pandas as pd
import sys
import h3
import os
import numpy as np

INPUT_FILE = "/opt/airflow/data/raw_datatourisme.csv"
OUTPUT_FILE = "/opt/airflow/data/clean_pois.csv"

# --- MAPPING CATÉGORIES (Copié de votre version originale) ---
SUB_TO_MAIN = {
    0: 8, 1: 18, 2: 18, 3: 0, 4: 13, 5: 9, 6: 0, 7: 9, 8: 10, 9: 13, 
    10: 8, 11: 7, 12: 9, 13: 8, 14: 9, 15: 3, 16: 12, 17: 3, 18: 0, 
    19: 6, 20: 13, 21: 6, 22: 15, 23: 4, 24: 3, 25: 10, 26: 8, 27: 1, 
    28: 10, 29: 13, 30: 0, 31: 0, 32: 16, 33: 18, 34: 8, 35: 18, 36: 13, 
    37: 10, 38: 15, 39: 0, 40: 3, 41: 18, 42: 2, 43: 13, 44: 10, 45: 14, 
    46: 10, 47: 0, 48: 13, 49: 18, 50: 13, 51: 10, 52: 17, 53: 6, 54: 10, 
    55: 2, 56: 5, 57: 9, 58: 18, 59: 11, 60: 0, 61: 13, 62: 2, 63: 13, 
    64: 6, 65: 8
}

KEYWORD_MAP = {
    'restauration rapide': 0, 'fast food': 0, 'snack': 0,
    'chateau': 1, 'fort': 1, 'citadelle': 1,
    'eglise': 2, 'abbaye': 2, 'cathedrale': 2, 'chapelle': 2, 'religieux': 2,
    'plage': 3, 'mer': 3, 'littoral': 3,
    'tennis': 4, 'raquette': 4, 'artisan': 5,
    'lac': 6, 'etang': 6, 'humide': 6,
    'supermarche': 7, 'commerce': 7, 'boutique': 7,
    'bibliotheque': 8, 'mediatheque': 8,
    'bowling': 9, 'laser': 9, 'indoor': 9,
    'producteur': 10, 'ferme': 10, 'brocante': 12, 'antiquite': 12,
    'restaurant': 13, 'auberge': 13, 'marche': 14,
    'parc': 15, 'jardin': 15, 'foret': 18, 'naturel': 18,
    'pique-nique': 19, 'cheval': 20, 'equestre': 20,
    'bus touristique': 22, 'train touristique': 22,
    'zoo': 24, 'animal': 24, 'spectacle': 25,
    'cafe': 26, 'bar': 26, 'conference': 28,
    'nautique': 29, 'voile': 29, 'paysage': 30, 'panorama': 30,
    'montagne': 31, 'ouvrage': 33, 'pont': 33, 'produit local': 34,
    'golf': 36, 'musee': 37, 'expo': 37, 'telepherique': 38,
    'cascade': 39, 'eau vive': 39, 'aire de jeux': 40, 'thermal': 42,
    'stade': 43, 'sport collectif': 43, 'cinema': 44,
    'geologie': 47, 'grotte': 47, 'sport mecanique': 48, 'karting': 48,
    'patrimoine civil': 49, 'randonnee': 50, 'outdoor': 50,
    'concert': 51, 'musique': 51, 'fete': 53, 'tradition': 53,
    'festival': 54, 'soin': 55, 'bien-etre': 55, 'foire': 57, 'salon': 57,
    'cimetiere': 58, 'memorial': 58, 'ski': 61, 'hiver': 61,
    'thalasso': 62, 'balneo': 62, 'accrobranche': 63, 'aventure': 63,
    'defile': 64, 'parade': 64, 'vin': 65, 'spiritueux': 65, 'cave': 65
}

def determine_categories(nom, desc):
    # Sécurisation si les champs sont vides
    nom = str(nom) if pd.notna(nom) else ""
    desc = str(desc) if pd.notna(desc) else ""
    text = (nom + " " + desc).lower()
    
    for keyword, sub_id in KEYWORD_MAP.items():
        if keyword in text:
            return sub_id, SUB_TO_MAIN.get(sub_id, 7)
    return 11, 7

def calculate_h3(lat, lon, res):
    try:
        return h3.latlng_to_cell(float(lat), float(lon), res)
    except:
        return None

def transform():
    if not os.path.exists(INPUT_FILE):
        print("❌ Pas de fichier source (raw_datatourisme.csv)"); sys.exit(1)
    
    print("⚙️ Transformation en cours (Depuis CSV)...")
    
    # 1. Lecture du CSV généré par l'étape 1
    # On force les types pour éviter les erreurs de lecture
    df = pd.read_csv(INPUT_FILE, dtype={'addr_cp': str, 'latitude': float, 'longitude': float})
    
    print(f"   📊 {len(df)} POIs chargés.")
    
    # Suppression des lignes sans GPS
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"   🧹 {len(df)} POIs avec GPS valide.")

    # 2. Application des Catégories
    print("   🏷️ Calcul des catégories...")
    cats = df.apply(lambda x: determine_categories(x['nom_du_poi'], x['description']), axis=1)
    # zip(*cats) permet de séparer les tuples (sub_id, main_id) en deux colonnes
    df['sub_category_id'], df['main_category_id'] = zip(*cats)

    # 3. Calcul H3
    print("   hex Calcul des index H3...")
    for r in [6, 7, 8, 9]:
        df[f'h3_r{r}'] = df.apply(lambda x: calculate_h3(x['latitude'], x['longitude'], r), axis=1)

    # 4. Initialisation des scores (0.0)
    scores = ['density_commune_norm', 'diversity_commune_norm', 'popularity_norm', 
              'proximity_commune_norm', 'category_weight_norm', 'opening_score_norm', 'final_score']
    for s in scores: 
        df[s] = 0.0
    
    df['contacts_du_poi'] = ''
    df['itineraire'] = ''
    
    # 5. Sauvegarde
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Transformation terminée. Fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    transform()