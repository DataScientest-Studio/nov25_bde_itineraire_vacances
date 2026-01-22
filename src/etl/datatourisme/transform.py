import pandas as pd
import h3
import random
from src.config.settings import RAW_DIR, PROCESSED_DIR
from datetime import datetime

def get_h3_index(lat, lon, resolution=8):
    try:
        # Syntax H3 version 4+
        return h3.latlng_to_cell(lat, lon, resolution)
    except:
        return None

def transform_datatourisme(file_path=None):
    print("🔄 Début de la Transformation (Logique 'Itinéraire')...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Récupération du fichier brut
    if not file_path:
        files = list((RAW_DIR / "datatourisme").glob("*.csv"))
        if not files: raise FileNotFoundError("Aucun fichier RAW trouvé.")
        file_path = max(files, key=lambda f: f.stat().st_mtime)

    # 2. Chargement (Optimisé pour les gros fichiers)
    print(f"📖 Lecture de : {file_path}")
    # On lit les colonnes standards DataTourisme (adapter selon le CSV réel)
    df = pd.read_csv(file_path, sep=',', on_bad_lines='skip', low_memory=False)
    
    # 3. Renommage et Sélection des colonnes
    # Note: Les CSV DataTourisme changent parfois. On mappe les noms probables.
    rename_map = {
        'Nom_du_POI': 'libelle',
        'Description_commerciale': 'description',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Adresse_postale': 'adresse_full',
        'Code_postal_et_commune': 'cp_ville',
        'Categories_de_POI': 'categories_raw'
    }
    # Si les colonnes n'existent pas, Pandas les ignorera dans le rename, on filtre après.
    df = df.rename(columns=rename_map)
    
    # Vérification colonnes minimales
    required = ['libelle', 'latitude', 'longitude']
    if not all(col in df.columns for col in required):
        print(f"⚠️ Colonnes manquantes dans le CSV. Colonnes dispo : {df.columns}")
        # Fallback pour le CSV "Simplifié" si noms différents
        rename_map_v2 = {
            'Nom offre': 'libelle', 'Latitude': 'latitude', 'Longitude': 'longitude',
            'Adresse 1': 'adresse_full', 'Code postal': 'code_postal', 'Commune': 'ville'
        }
        df = df.rename(columns=rename_map_v2)

    # 4. 🧹 FILTRAGE ILE-DE-FRANCE
    # On extrait le code postal s'il est manquant ou mixé
    if 'code_postal' not in df.columns and 'cp_ville' in df.columns:
        df['code_postal'] = df['cp_ville'].astype(str).str.extract(r'^(\d{5})')
    
    # On garde uniquement les départements IDF (75, 77, 78, 91, 92, 93, 94, 95)
    print("📍 Filtrage Île-de-France...")
    df['dept'] = df['code_postal'].astype(str).str[:2]
    idf_depts = ['75', '77', '78', '91', '92', '93', '94', '95']
    df = df[df['dept'].isin(idf_depts)].copy()
    
    print(f"📊 POIs restants après filtrage IDF : {len(df)}")

    # 5. 🌍 Enrichissement H3 & PostGIS
    print("⬡ Calcul des index H3...")
    df['h3_index'] = df.apply(lambda x: get_h3_index(x['latitude'], x['longitude']), axis=1)
    
    # Préparation géométrie WKT pour PostGIS (POINT(lon lat))
    df['geom_wkt'] = df.apply(lambda x: f"POINT({x['longitude']} {x['latitude']})", axis=1)

    # 6. 🏷️ Catégorisation (Simplifiée)
    # On crée des catégories par défaut si manquantes
    df['main_category'] = 'Découverte' # Valeur par défaut
    df['sub_category'] = 'Lieu touristique'
    
    # Exemple de règle simple (à enrichir selon le contenu de 'categories_raw')
    if 'categories_raw' in df.columns:
        df.loc[df['categories_raw'].str.contains('Musée', case=False, na=False), 'main_category'] = 'Culture'
        df.loc[df['categories_raw'].str.contains('Musée', case=False, na=False), 'sub_category'] = 'Musée'
        df.loc[df['categories_raw'].str.contains('Parc', case=False, na=False), 'main_category'] = 'Nature'

    # 7. ⭐ Scoring (Placeholders 0-1 comme demandé)
    df['density_h3_r8'] = [random.uniform(0.1, 0.9) for _ in range(len(df))]
    df['diversity_h3_r8'] = [random.uniform(0.1, 0.9) for _ in range(len(df))]
    df['popularity'] = [random.uniform(0, 500) for _ in range(len(df))] # Popularité brute
    
    # Normalisation du Final Score (0-1)
    df['final_score'] = (df['density_h3_r8'] + df['diversity_h3_r8']) / 2

    # 8. Sauvegarde
    output_path = PROCESSED_DIR / f"processed_vacances_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # On garde les colonnes utiles pour le LOAD
    cols_export = [
        'libelle', 'description', 'latitude', 'longitude', 'geom_wkt', 
        'h3_index', 'density_h3_r8', 'diversity_h3_r8', 'popularity', 'final_score',
        'main_category', 'sub_category', 
        'adresse_full', 'code_postal', 'ville'
    ]
    # On ne garde que les colonnes qui existent vraiment
    final_cols = [c for c in cols_export if c in df.columns]
    
    df[final_cols].to_csv(output_path, index=False)
    print(f"✅ Transformation terminée : {output_path} ({len(df)} POIs)")
    return str(output_path)

if __name__ == "__main__":
    transform_datatourisme()