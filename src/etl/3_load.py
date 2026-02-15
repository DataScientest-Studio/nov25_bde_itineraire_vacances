import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

INPUT_FILE = "/opt/airflow/data/clean_pois.csv"
DB_URI = "postgresql+psycopg2://vacances_user:vacances_pass@postgres-vacances:5432/vacances"


def truncate_text(text, length=255):
    if pd.isna(text) or text == "":
        return None
    return str(text)[:length]


def load():
    if not os.path.exists(INPUT_FILE):
        print("❌ Fichier clean introuvable.")
        sys.exit(1)

    engine = create_engine(DB_URI)
    print("📥 Lecture CSV...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Nettoyage types
    for col in [
        "latitude",
        "longitude",
        "density_commune_norm",
        "diversity_commune_norm",
        "popularity_norm",
        "proximity_commune_norm",
        "category_weight_norm",
        "opening_score_norm",
        "final_score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Conversion IDs catégories
    df["main_category_id"] = (
        pd.to_numeric(df["main_category_id"], errors="coerce").fillna(7).astype(int)
    )
    df["sub_category_id"] = (
        pd.to_numeric(df["sub_category_id"], errors="coerce").fillna(11).astype(int)
    )

    print(f"✅ {len(df)} POIs prêts.")

    # On ne vide que POI et ADRESSE (catégories intactes)
    print("🧹 Reset POI/ADRESSE...")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE poi, adresse RESTART IDENTITY CASCADE;"))

    print("💾 Insertion...")
    count = 0
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for _, row in df.iterrows():
                # 1. Insertion Adresse
                res = conn.execute(
                    text(
                        """
                    INSERT INTO adresse (rue, code_postal, commune)
                    VALUES (:r, :cp, :c)
                    RETURNING id
                """
                    ),
                    {
                        "r": truncate_text(row.get("addr_rue")),
                        "cp": truncate_text(row.get("addr_cp"), 20),
                        "c": truncate_text(row.get("addr_commune"), 100),
                    },
                )
                addr_id = res.fetchone()[0]

                # 2. Insertion POI (Avec vos noms exacts)
                # Note: poi_id est AUTO-GENERÉ par Serial, donc on ne l'insère pas
                conn.execute(
                    text(
                        """
                    INSERT INTO poi (
                        nom_du_poi, latitude, longitude, description,
                        main_category_id, sub_category_id, adresse_id,
                        itineraire,
                        contact_mail, contact_phone, contact_website,
                        h3_r6, h3_r7, h3_r8, h3_r9,
                        density_commune_norm, diversity_commune_norm, popularity_norm,
                        proximity_commune_norm, category_weight_norm, opening_score_norm, final_score
                    ) VALUES (
                        :nom, :lat, :lon, :desc,
                        :main, :sub, :aid,
                        :itin,
                        :mail, :phone, :web,   
                        :h6, :h7, :h8, :h9,
                        :den, :div, :pop, :prox, :cat_w, :open, :final
                    )
                """
                    ),
                    {
                        "nom": truncate_text(row["nom_du_poi"]),
                        "lat": row["latitude"],
                        "lon": row["longitude"],
                        "desc": row["description"],
                        "main": row["main_category_id"],
                        "sub": row["sub_category_id"],
                        "aid": addr_id,
                        "itin": row["itineraire"],
                        "mail": row["contact_mail"],
                        "phone": row["contact_phone"],
                        "web": row["contact_website"],
                        "h6": row["h3_r6"],
                        "h7": row["h3_r7"],
                        "h8": row["h3_r8"],
                        "h9": row["h3_r9"],
                        "den": row["density_commune_norm"],
                        "div": row["diversity_commune_norm"],
                        "pop": row["popularity_norm"],
                        "prox": row["proximity_commune_norm"],
                        "cat_w": row["category_weight_norm"],
                        "open": row["opening_score_norm"],
                        "final": row["final_score"],
                    },
                )
                count += 1
                if count % 5000 == 0:
                    print(f"   ... {count} insérés")

            trans.commit()
            print(f"✅ {count} POIs insérés.")
        except Exception as e:
            trans.rollback()
            print(f"❌ Erreur: {e}")
            sys.exit(1)


if __name__ == "__main__":
    load()
