import requests
import sys
import os
import shutil

DATATOURISME_URL = "https://diffuseur.datatourisme.fr/webservice/613d5769ae6bcd51acd2936eeabff53b/0f025bf9-d4b4-4abb-8600-b574c35fe273"
OUTPUT_FILE = "/opt/airflow/data/raw_datatourisme.csv"

def extract():
    print(f"📥 Démarrage téléchargement (Version légère 10k)...")
    print(f"🔗 Source : {DATATOURISME_URL}")

    # Création du dossier si nécessaire
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        with requests.Session() as s:
            with s.get(DATATOURISME_URL, headers=headers, stream=True, timeout=600) as r:
                r.raise_for_status()
                
                total_size = 0
                with open(OUTPUT_FILE, 'wb') as f:
                    print("⏳ Téléchargement...", end="", flush=True)
                    
                    # ✅ CORRECTION CRITIQUE : Utiliser iter_content
                    # Cela décompresse automatiquement le flux GZIP pour avoir un vrai ZIP/CSV à l'arrivée
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
            
            print(f"\n✅ Terminé ! Fichier sauvegardé : {OUTPUT_FILE}")
            print(f"📦 Taille : {total_size / (1024*1024):.2f} Mo")

    except Exception as e:
        print(f"\n❌ Erreur extract : {e}")
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        sys.exit(1)

if __name__ == "__main__":
    extract()