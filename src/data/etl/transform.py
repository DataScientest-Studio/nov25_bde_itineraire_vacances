import polars as pl
import re
from pathlib import Path
import json

ROOT = Path(__file__).parent # parent directory of the script
INPUT_DIR = ROOT / "config"
input_categories_path = INPUT_DIR / "categories.json"
input_uri_mapping_path = INPUT_DIR / "classes_fr.csv"

TYPES_A_IGNORER = [
    "PlaceOfInterest",
    "PointOfInterest",
    "LocalBusiness",
    "Organization",
    "Agent",
    "Product",
    "Thing",
    "Place",
    "OrderedList",
    "CyclingTour",
    "CycleRouteTheme",
    "FluvialTour",
    "HorseTour",
    "RoadTour",
    "UnderwaterRoute",
    "WalkingTour"
]


# ---------------------------------------------------------
# 1) Normalisation des noms de colongitudenes
# ---------------------------------------------------------
def normalize_column_name(col: str) -> str:
    """
    Convertit un nom de colongitudene :
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
# 2) Strip de toutes les colongitudenes string
# ---------------------------------------------------------
def strip_all_string_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Applique un strip() sur toutes les colongitudenes de type Utf8.
    """
    return df.with_columns([
        pl.col(pl.Utf8).str.strip_chars()

    ])

# ---------------------------------------------------------
# 3) Suppression des doublongitudes
# ---------------------------------------------------------
def drop_duplicates(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.unique()


# ---------------------------------------------------------
# 4) Split de "code_postal_commune"
# ---------------------------------------------------------
def split_code_postal_commune(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Suppose une colongitudene 'code_postal_et_commune' de type :
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
# ---------------------------------------------------------
# charger le mapping csv (uri -> label)
def load_uri_mapping(path: str = input_uri_mapping_path) -> pl.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split uniquement sur la première virgule
            uri, label = line.split(",", 1)

            # Nettoyage de l’URI
            uri_clean = uri.replace("<", "").replace(">", "")
            type_clean = uri_clean.split("#")[-1]

            # Nettoyage du label
            label_clean = (
                label
                .strip()
                .strip('"')
                .replace("<", "")
                .replace(">", "")
            )

            # Supprimer tout ce qui ressemble à un URI dans le label
            if "http" in label_clean:
                label_clean = label_clean.split("http")[0].rstrip(", ")

            rows.append({
                "type_clean": type_clean,
                "Label": label_clean
            })

    return pl.DataFrame(rows)


# charger categories.json (label -> main/sub/type)
def load_category_hierarchy(path: str = input_categories_path) -> pl.DataFrame:
    """
    Transforme le JSON hiérarchique en DataFrame platitude :
    type_principal → main_category, sub_category
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for main_cat, subcats in data.items():
        for sub_cat, labels in subcats.items():
            for label in labels:
                rows.append({
                    "type_principal": label,
                    "main_category": main_cat,
                    "sub_category": sub_cat
                })

    return pl.DataFrame(rows)


#  Extraction des types depuis categories_de_poi
def extract_types(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns([
        pl.col("categories_de_poi")
        .str.extract_all(r"#([A-Za-z0-9]+)")   # extrait après #
        .list.eval(
            pl.element()
            .str.replace_all(r"[^A-Za-z0-9]", "")  # supprime #, <, >, ', ", espaces
        )
        .list.eval(
            pl.element().filter(~pl.element().is_in(TYPES_A_IGNORER))
        )
        .alias("types_list")
    ])


# extraction du type principal
def extract_type_principal(df: pl.LazyFrame, mapping_df: pl.DataFrame) -> pl.LazyFrame:
    exploded = df.explode("types_list")

    joined = exploded.join(
        mapping_df,
        left_on="types_list",
        right_on="type_clean",
        how="left"
    )

    aggregated = joined.group_by(df.columns).agg([
        pl.col("Label").drop_nulls().first().alias("type_principal")
    ])

    return aggregated

# enrichissement avec main_category / sub_category
def enrich_with_categories(df: pl.LazyFrame, cat_df: pl.DataFrame) -> pl.LazyFrame:
    return df.join(
        cat_df,
        on="type_principal",
        how="left"
    )


# pipeline complet pour le mapping
def apply_full_mapping(df: pl.LazyFrame) -> pl.LazyFrame:
    mapping_df = load_uri_mapping()
    cat_df = load_category_hierarchy()

    df = extract_types(df)
    df = extract_type_principal(df, mapping_df)
    df = enrich_with_categories(df, cat_df)
    
    # Rajout colongitudene itinéraire Tue /False
    df = df.with_columns([
        (pl.col("sub_category") != "unknown").alias("itineraire")
    ])

    return df



# ============================================================
# 8) Nettoyage final
# ============================================================
def drop_null_categories(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(
        ~(pl.col("main_category").is_null() & pl.col("sub_category").is_null())
    )

def final_cleanup(df: pl.LazyFrame) -> pl.LazyFrame:
    # Supprimer les lignes où main_category est NULL
    df = df.filter(pl.col("main_category").is_not_null())

    return df

def safe_rename(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {
        "adresse_postale": "adresse",
    }

    existing = {old: new for old, new in rename_map.items() if old in df.columns}
    return df.rename(existing)


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

    # MAPPING
    df = apply_full_mapping(df)

    # NETTOYAGE FINAL
    print('avant drop main et sub', len(df))
    df = drop_null_categories(df)
    print('apres drop main et sub', len(df))
    df = final_cleanup(df)
    print('apres final cleanup', len(df))
    df = safe_rename(df)

    return df