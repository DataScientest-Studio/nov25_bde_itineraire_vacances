import json
import os
import sys
import zipfile

import h3
import numpy as np
import pandas as pd

INPUT_FILE = "/opt/airflow/data/raw_datatourisme.csv"
OUTPUT_FILE = "/opt/airflow/data/clean_pois.csv"
BATCH_SIZE = 5000

# --- MAPPING CATÉGORIES
# Sub ID -> Main ID
SUB_TO_MAIN = {
    0: 8,
    1: 18,
    2: 18,
    3: 0,
    4: 13,
    5: 9,
    6: 0,
    7: 9,
    8: 10,
    9: 13,
    10: 8,
    11: 7,
    12: 9,
    13: 8,
    14: 9,
    15: 3,
    16: 12,
    17: 3,
    18: 0,
    19: 6,
    20: 13,
    21: 6,
    22: 15,
    23: 4,
    24: 3,
    25: 10,
    26: 8,
    27: 1,
    28: 10,
    29: 13,
    30: 0,
    31: 0,
    32: 16,
    33: 18,
    34: 8,
    35: 18,
    36: 13,
    37: 10,
    38: 15,
    39: 0,
    40: 3,
    41: 18,
    42: 2,
    43: 13,
    44: 10,
    45: 14,
    46: 10,
    47: 0,
    48: 13,
    49: 18,
    50: 13,
    51: 10,
    52: 17,
    53: 6,
    54: 10,
    55: 2,
    56: 5,
    57: 9,
    58: 18,
    59: 11,
    60: 0,
    61: 13,
    62: 2,
    63: 13,
    64: 6,
    65: 8,
}

KEYWORD_MAP = {
    "restauration rapide": 0,
    "fast food": 0,
    "snack": 0,
    "chateau": 1,
    "fort": 1,
    "citadelle": 1,
    "eglise": 2,
    "abbaye": 2,
    "cathedrale": 2,
    "chapelle": 2,
    "religieux": 2,
    "plage": 3,
    "mer": 3,
    "littoral": 3,
    "tennis": 4,
    "raquette": 4,
    "artisan": 5,
    "lac": 6,
    "etang": 6,
    "humide": 6,
    "supermarche": 7,
    "commerce": 7,
    "boutique": 7,
    "bibliotheque": 8,
    "mediatheque": 8,
    "bowling": 9,
    "laser": 9,
    "indoor": 9,
    "producteur": 10,
    "ferme": 10,
    "brocante": 12,
    "antiquite": 12,
    "restaurant": 13,
    "auberge": 13,
    "marche": 14,
    "parc": 15,
    "jardin": 15,
    "foret": 18,
    "naturel": 18,
    "pique-nique": 19,
    "cheval": 20,
    "equestre": 20,
    "bus touristique": 22,
    "train touristique": 22,
    "zoo": 24,
    "animal": 24,
    "spectacle": 25,
    "cafe": 26,
    "bar": 26,
    "conference": 28,
    "nautique": 29,
    "voile": 29,
    "paysage": 30,
    "panorama": 30,
    "montagne": 31,
    "ouvrage": 33,
    "pont": 33,
    "produit local": 34,
    "golf": 36,
    "musee": 37,
    "expo": 37,
    "telepherique": 38,
    "cascade": 39,
    "eau vive": 39,
    "aire de jeux": 40,
    "thermal": 42,
    "stade": 43,
    "sport collectif": 43,
    "cinema": 44,
    "geologie": 47,
    "grotte": 47,
    "sport mecanique": 48,
    "karting": 48,
    "patrimoine civil": 49,
    "randonnee": 50,
    "outdoor": 50,
    "concert": 51,
    "musique": 51,
    "fete": 53,
    "tradition": 53,
    "festival": 54,
    "soin": 55,
    "bien-etre": 55,
    "foire": 57,
    "salon": 57,
    "cimetiere": 58,
    "memorial": 58,
    "ski": 61,
    "hiver": 61,
    "thalasso": 62,
    "balneo": 62,
    "accrobranche": 63,
    "aventure": 63,
    "defile": 64,
    "parade": 64,
    "vin": 65,
    "spiritueux": 65,
    "cave": 65,
}


def determine_categories(nom, desc):
    text = (str(nom) + " " + str(desc)).lower()
    for keyword, sub_id in KEYWORD_MAP.items():
        if keyword in text:
            return sub_id, SUB_TO_MAIN.get(sub_id, 7)
    return 11, 7


def calculate_h3(lat, lon, res):
    try:
        if pd.isna(lat) or pd.isna(lon):
            return None
        return h3.latlng_to_cell(float(lat), float(lon), res)
    except:
        return None


def extract_nested(row, keys):
    try:
        roots = [
            row.get("isLocatedAt"),
            row.get("hasLocation"),
            row.get("schema:address"),
        ]
        for root in roots:
            if not root:
                continue
            if isinstance(root, list) and len(root) > 0:
                root = root[0]
            if isinstance(root, dict):
                addr = root.get("schema:address") or root.get("address") or root
                if isinstance(addr, list) and len(addr) > 0:
                    addr = addr[0]
                if isinstance(addr, dict):
                    for k, v in addr.items():
                        for h in keys:
                            if h in k:
                                return v
    except:
        pass
    return None


def extract_geo(row, target):
    try:
        loc = row.get("isLocatedAt") or row.get("hasLocation")
        if isinstance(loc, list) and len(loc) > 0:
            loc = loc[0]
        if isinstance(loc, dict):
            geo = loc.get("schema:geo") or loc.get("geo")
            if isinstance(geo, list) and len(geo) > 0:
                geo = geo[0]
            if isinstance(geo, dict):
                return geo.get(f"schema:{target}") or geo.get(target)
    except:
        pass
    return None


def process_batch(data_list, first_batch):
    if not data_list:
        return
    df = pd.json_normalize(data_list)
    df_clean = pd.DataFrame()

    # 1. INFO BASE
    def get_col(candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    # Mapping Nom -> nom_du_poi
    nom = get_col(["rdfs:label.fr", "Nom_du_POI", "nom"])
    df_clean["nom_du_poi"] = (
        nom.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        if nom is not None
        else "Inconnu"
    )

    desc = get_col(["rdfs:comment.fr", "Description"])
    df_clean["description"] = (
        desc.apply(lambda x: str(x)[0:500] if isinstance(x, (str, list)) else "")
        if desc is not None
        else ""
    )

    # 2. ADRESSE (Pour l'insertion dans la table adresse d'abord)
    df_clean["addr_rue"] = df.apply(
        lambda x: extract_nested(x, ["streetAddress"]), axis=1
    ).fillna("")
    df_clean["addr_cp"] = df.apply(
        lambda x: extract_nested(x, ["postalCode"]), axis=1
    ).fillna("")
    df_clean["addr_commune"] = df.apply(
        lambda x: extract_nested(x, ["addressLocality", "locality"]), axis=1
    ).fillna("")

    # 3. GPS
    df_clean["latitude"] = pd.to_numeric(
        df.apply(lambda x: extract_geo(x, "latitude"), axis=1), errors="coerce"
    )
    df_clean["longitude"] = pd.to_numeric(
        df.apply(lambda x: extract_geo(x, "longitude"), axis=1), errors="coerce"
    )

    # 4. CATEGORIES
    cats = df_clean.apply(
        lambda x: determine_categories(x["nom_du_poi"], x["description"]), axis=1
    )
    df_clean["sub_category_id"] = cats.apply(lambda x: x[0])
    df_clean["main_category_id"] = cats.apply(lambda x: x[1])

    # 5. H3 & SCORES (Nommés exactement comme votre demande)
    for r in [6, 7, 8, 9]:
        df_clean[f"h3_r{r}"] = df_clean.apply(
            lambda x: calculate_h3(x["latitude"], x["longitude"], r), axis=1
        )

    scores = [
        "density_commune_norm",
        "diversity_commune_norm",
        "popularity_norm",
        "proximity_commune_norm",
        "category_weight_norm",
        "opening_score_norm",
        "final_score",
    ]
    for s in scores:
        df_clean[s] = 0.0

    df_clean["contacts_du_poi"] = ""
    df_clean["itineraire"] = ""

    df_clean = df_clean.dropna(subset=["latitude", "longitude"])

    mode = "w" if first_batch else "a"
    header = True if first_batch else False
    df_clean.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)


def transform():
    if not os.path.exists(INPUT_FILE):
        print("❌ Pas de fichier source")
        sys.exit(1)

    print("⚙️ Transformation (Structure Finale)...")
    try:
        with zipfile.ZipFile(INPUT_FILE, "r") as z:
            files = [f for f in z.namelist() if f.endswith(".json")]
            batch = []
            is_first = True
            for i, f in enumerate(files):
                with z.open(f) as jf:
                    try:
                        d = json.load(jf)
                        if isinstance(d, list):
                            batch.extend(d)
                        else:
                            batch.append(d)
                    except:
                        continue
                if len(batch) >= BATCH_SIZE:
                    process_batch(batch, is_first)
                    batch = []
                    is_first = False
            if batch:
                process_batch(batch, is_first)
        print("✅ Transformation terminée.")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    transform()
