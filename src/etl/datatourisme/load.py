import pandas as pd
from src.config.settings import PROCESSED_DIR
from src.utils.db_connector import Database
from sqlalchemy import text

def load_datatourisme():
    print("💾 Début du Chargement (Architecture Staging)...")
    
    # 1. Trouver le fichier
    files = list(PROCESSED_DIR.glob("processed_vacances_*.csv"))
    if not files: raise FileNotFoundError("Aucun fichier transformé trouvé.")
    file_path = max(files, key=lambda f: f.stat().st_mtime)
    
    df = pd.read_csv(file_path)
    print(f"📂 Fichier chargé : {len(df)} lignes")
    
    db = Database()
    
    # 2. Chargement dans une table TEMPORAIRE (Staging)
    # Cela évite de gérer les Foreign Keys en Python (trop lent)
    print("⏳ Création de la table de staging...")
    df.to_sql('stg_poi', db.engine, if_exists='replace', index=False)
    
    # 3. Exécution des requêtes SQL complexes (ELT)
    with db.engine.begin() as conn:
        print("🔧 Mise à jour des référentiels (Catégories & Adresses)...")
        
        # A. Insérer les Main Categories manquantes
        conn.execute(text("""
            INSERT INTO main_categorie (libelle)
            SELECT DISTINCT main_category FROM stg_poi
            WHERE main_category IS NOT NULL
            ON CONFLICT DO NOTHING;
        """))
        
        # B. Insérer les Sub Categories manquantes
        conn.execute(text("""
            INSERT INTO sub_categorie (libelle, main_category_id)
            SELECT DISTINCT s.sub_category, m.id
            FROM stg_poi s
            JOIN main_categorie m ON s.main_category = m.libelle
            ON CONFLICT DO NOTHING;
        """))
        
        # C. Insérer les Adresses
        # (On suppose ici qu'une adresse est unique par sa chaîne complète)
        conn.execute(text("""
            INSERT INTO adresse (adresse_complete, code_postal, ville)
            SELECT DISTINCT adresse_full, code_postal, ville 
            FROM stg_poi
            ON CONFLICT DO NOTHING;
        """))
        
        print("🚀 Upsert des POIs...")
        
        # D. Insérer les POIs finaux avec les liens (JOIN)
        # On utilise geom_wkt pour créer le point PostGIS
        query_poi = """
            INSERT INTO poi (
                libelle, description, latitude, longitude, geom, 
                h3_index, density_h3_r8, diversity_h3_r8, popularity, final_score,
                main_category_id, sub_category_id, adresse_id, source_id
            )
            SELECT 
                s.libelle, s.description, s.latitude, s.longitude, 
                ST_GeomFromText(s.geom_wkt, 4326), -- Conversion PostGIS
                s.h3_index, s.density_h3_r8, s.diversity_h3_r8, s.popularity, s.final_score,
                m.id, sub.id, a.id, 
                CONCAT(s.latitude, '_', s.longitude) -- ID unique temporaire (lat_lon)
            FROM stg_poi s
            LEFT JOIN main_categorie m ON s.main_category = m.libelle
            LEFT JOIN sub_categorie sub ON s.sub_category = sub.libelle
            LEFT JOIN adresse a ON s.adresse_full = a.adresse_complete
            ON CONFLICT (source_id) DO UPDATE SET
                final_score = EXCLUDED.final_score,
                popularity = EXCLUDED.popularity;
        """
        conn.execute(text(query_poi))
        
        # Nettoyage
        conn.execute(text("DROP TABLE stg_poi;"))
        
    print("✅ Chargement Terminé ! Base de données à jour.")
    return "Success"

if __name__ == "__main__":
    load_datatourisme()