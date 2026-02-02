import requests
import os
import sys
import zipfile
import json
import csv
import time

# --- Configuration ---
APP_KEY = "0f025bf9-d4b4-4abb-8600-b574c35fe273"

URLS = {
    "bretagne": f"https://diffuseur.datatourisme.fr/webservice/3e9bc051440b15286cd66710d4ddd5d5/{APP_KEY}",
    "ile_de_france": f"https://diffuseur.datatourisme.fr/webservice/613d5769ae6bcd51acd2936eeabff53b/{APP_KEY}",
    "auvergne": f"https://diffuseur.datatourisme.fr/webservice/c010fb7faf7afeabb423d91aac388984/{APP_KEY}"
}

OUTPUT_FILE = "/opt/airflow/data/raw_datatourisme.csv"
TEMP_ZIP = "/tmp/temp_flux.zip"

def get_flattened_data(data, region_name):
    """Extrait ID, Nom, Desc, GPS et Adresse du JSON complexe."""
    try:
        # 1. Nom
        raw_label = data.get("rdfs:label")
        nom = "Sans nom"
        if isinstance(raw_label, dict):
            nom = raw_label.get("fr", ["Sans nom"])[0]
        elif isinstance(raw_label, str):
            nom = raw_label

        # 2. Description
        desc = ""
        has_desc = data.get("hasDescription")
        if has_desc and isinstance(has_desc, list) and len(has_desc) > 0:
            short_desc = has_desc[0].get("shortDescription")
            if isinstance(short_desc, dict):
                desc = short_desc.get("fr", [""])[0]
        if desc: desc = desc.replace('\n', ' ').replace('\r', '')

        # 3. Localisation (GPS & Adresse)
        lat, lon, street, cp, city = None, None, "", "", ""
        
        loc = data.get('isLocatedAt') or data.get('hasLocation')
        if loc:
            if isinstance(loc, list): loc = loc[0]
            if isinstance(loc, dict):
                # GPS
                geo = loc.get('schema:geo') or loc.get('geo')
                if geo:
                    if isinstance(geo, list): geo = geo[0]
                    if isinstance(geo, dict):
                        lat = geo.get('schema:latitude') or geo.get('latitude')
                        lon = geo.get('schema:longitude') or geo.get('longitude')
                
                # Adresse
                addr = loc.get('schema:address') or loc.get('address')
                if addr:
                    if isinstance(addr, list): addr = addr[0]
                    if isinstance(addr, dict):
                        street = addr.get('schema:streetAddress') or addr.get('streetAddress') or ""
                        cp = addr.get('schema:postalCode') or addr.get('postalCode') or ""
                        city = addr.get('schema:addressLocality') or addr.get('addressLocality') or ""

        return [data.get("@id", ""), nom, desc, lat, lon, street, cp, city, region_name]
    except Exception:
        return None

def download_file(url, target_path):
    headers = {'User-Agent': 'DataTourismeBot/1.0', 'Accept-Encoding': 'gzip, deflate'}
    with requests.get(url, headers=headers, stream=True, timeout=1200) as r:
        r.raise_for_status()
        total_size = 0
        with open(target_path, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=1024*1024)):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    if i % 10 == 0: print(f"      ⏳ {(total_size/1024/1024):.1f} Mo...", flush=True)
    return total_size

def extract():
    print(f"📥 Extraction (Mode CSV Flat avec GPS)...", flush=True)
    
    # Dossiers et Nettoyage
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)): os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Écriture En-tête CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["id", "nom_du_poi", "description", "latitude", "longitude", "addr_rue", "addr_cp", "addr_commune", "source_region"])

        for region_name, url in URLS.items():
            print(f"   ➡️ {region_name.upper()}...", flush=True)
            try:
                download_file(url, TEMP_ZIP)
                
                if not zipfile.is_zipfile(TEMP_ZIP):
                    print("      ❌ Pas un ZIP valide", flush=True); continue

                with zipfile.ZipFile(TEMP_ZIP, 'r') as z:
                    json_files = [f for f in z.namelist() if f.startswith('objects/') and f.endswith('.json')]
                    print(f"      📦 {len(json_files)} fichiers à traiter...", flush=True)
                    
                    count = 0
                    for json_file in json_files:
                        try:
                            with z.open(json_file) as f_json:
                                row = get_flattened_data(json.load(f_json), region_name)
                                if row:
                                    writer.writerow(row)
                                    count += 1
                        except: continue
                    print(f"      ✅ {count} lignes ajoutées.", flush=True)

            except Exception as e:
                print(f"      ❌ Erreur: {e}", flush=True)
            finally:
                if os.path.exists(TEMP_ZIP): os.remove(TEMP_ZIP)

if __name__ == "__main__":
    extract()