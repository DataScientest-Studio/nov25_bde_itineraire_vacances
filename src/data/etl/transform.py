import polars as pl
import re
from pathlib import Path
import json

ROOT = Path(__file__).parent # parent directory of the script
INPUT_DIR = ROOT / "config"
input_path = INPUT_DIR / "categories.json"

# ---------------------------------------------------------
# 1) Normalisation des noms de colonnes
# ---------------------------------------------------------
def normalize_column_name(col: str) -> str:
    """
    Convertit un nom de colonne :
    - en minuscule
    - strip début/fin
    - remplace espaces par '_'
    - remplace caractères spéciaux simples
    """
    col = col.strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = col.replace("é", "e").replace("è", "e").replace("ê", "e")
    col = col.replace("à", "a").replace("ç", "c")
    return col


def rename_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    new_cols = {col: normalize_column_name(col) for col in df.columns}
    return df.rename(new_cols)


# ---------------------------------------------------------
# 2) Strip de toutes les colonnes string
# ---------------------------------------------------------
def strip_all_string_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Applique un strip() sur toutes les colonnes de type Utf8.
    """
    return df.with_columns([
        pl.col(pl.Utf8).str.strip_chars()

    ])


# ---------------------------------------------------------
# 3) Suppression des doublons
# ---------------------------------------------------------
def drop_duplicates(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.unique()


# ---------------------------------------------------------
# 4) Split de "code_postal_commune"
# ---------------------------------------------------------
def split_code_postal_commune(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Suppose une colonne 'code_postal_et_commune' de type :
    '75001 Paris'
    '13002 Marseille'
    etc.

    On extrait :
    - code_postal
    - commune
    - departement (les 2 premiers chiffres du code postal)
    """
    return df.with_columns([
        pl.col("code_postal_et_commune").str.split("#", inclusive=False).alias("tmp_split")
    ]).with_columns([
        pl.col("tmp_split").list.get(0).alias("code_postal"),
        pl.col("tmp_split").list.slice(1).list.join(" ").alias("commune"),
        pl.col("tmp_split").list.get(0).str.slice(0, 2).alias("departement")
    ]).drop("tmp_split")


# ---------------------------------------------------------
# 5) Extraction / nettoyage de "categories_poi"
# (placeholder : tu fourniras les règles)
# ---------------------------------------------------------
def clean_categories_poi(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Opérations en plusieurs étapes pour nettoyer la colonne catégories 
    """
    generic_types = [
        "PlaceOfInterest",
        "PointOfInterest",
        "LocalBusiness",
        "Organization",
        "Agent",
        "Product",
        "Thing",
        "Place",
        "OrderedList",
    ]

    df = (
        df
        # Split sur "|"
        .with_columns(
            pl.col("categories_de_poi").str.split("|").alias("types_raw")
        )
        # Extraire la partie après "#"
        .with_columns(
            pl.col("types_raw")
            .list.eval(
                pl.element()
                .str.replace("http://schema.org/", "")   # supprime le prefix schema.org
                .str.split("#").list.last()              # extrait la partie après #
            )
            .alias("types_clean")
        )
        # Filtrer les types génériques
        .with_columns(
            pl.col("types_clean")
            .list.eval(pl.element().filter(~pl.element().is_in(generic_types)))
            .alias("types_filtered")
        )
        # Sélectionner le type le plus spécifique
        .with_columns(
            pl.col("types_filtered").list.last().alias("type_principal")
        )
    )
    
    return df

# ============================================================
# 6) Chargement du mapping inverse
# ============================================================
def load_mapping(path: str = input_path) -> pl.LazyFrame:
    """
    mapping.json doit contenir une liste d'objets du type :
    [
        {"type": "restaurant", "categorie": "food"},
        {"type": "hotel", "categorie": "lodging"}
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rows = []
    for main_cat, subcats in data.items():
        for sub_cat, types in subcats.items():
            for t in types:
                rows.append((t, main_cat, sub_cat))

    return pl.DataFrame(rows, schema=["type", "main_category", "sub_category"])


# ============================================================
# 7) Application du mapping inverse (JOIN LAZY)
# ============================================================
def apply_mapping(df: pl.LazyFrame, mapping_df: pl.DataFrame) -> pl.LazyFrame:
    return df.join(
        mapping_df,
        left_on="type_principal",
        right_on="type",
        how="left"
    )

# ============================================================
# 8) Nettoyage final
# ============================================================
def final_cleanup(df: pl.LazyFrame) -> pl.LazyFrame:
    cols_to_drop = [
        "types_raw", "types_clean", "types_filtered", "type",
        "categories_de_poi", "code_postal_et_commune",
        "covid19_mesures_specifiques", "createur_de_la_donnee", "sit_diffuseur"
    ]

    # Supprimer les lignes où main_category est NULL
    df = df.filter(pl.col("main_category").is_not_null())

    # Supprimer uniquement les colonnes existantes
    cols_existing = [c for c in cols_to_drop if c in df.columns]
    if cols_existing:
        df = df.drop(cols_existing)

    return df


# ---------------------------------------------------------
# 9) Pipeline complet
# ---------------------------------------------------------
def transform(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Pipeline complet des transformations.
    """
    df = rename_columns(df)
    df = strip_all_string_columns(df)
    df = drop_duplicates(df)

    # Split code postal / commune / département
    if "code_postal_et_commune" in df.columns:
        df = split_code_postal_commune(df)

    # Nettoyage categories_poi (placeholder)
    if "categories_de_poi" in df.columns:
        df = clean_categories_poi(df)

    # MAPPING
    mapping_df = load_mapping()
    df = apply_mapping(df, mapping_df)

    # NETTOYAGE FINAL
    #df = final_cleanup(df)

    return df