import polars as pl


def build_category_tables(df: pl.DataFrame):
    df_main_category = (
        df.select("main_category")
          .unique()
          .with_row_index("main_category_id")
    )

    df = df.join(df_main_category, on="main_category", how="left")

    df_sub_category = (
        df.select(["sub_category", "main_category_id"])
          .unique()
          .with_row_index("sub_category_id")
    )

    df = df.join(
        df_sub_category,
        on=["sub_category", "main_category_id"],
        how="left"
    )

    return df_main_category, df_sub_category, df


def build_poi_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ajoute un identifiant unique pour chaque POI.
    """
    return df.with_row_index("poi_id")


def build_adresse_table(df: pl.DataFrame):
    """
    Construit la table adresse avec un adresse_id
    et ajoute adresse_id dans le DataFrame principal.
    """

    # On part du df qui contient déjà poi_id
    df_adresse = (
        df.select([
            "poi_id",
            "adresse",
            "code_postal",
            "commune",
            "departement",
            "region",
        ])
        .unique()                      # au cas où
        .with_row_index("adresse_id")  # PK adresse
    )

    # On réinjecte adresse_id dans le df principal
    df = df.join(df_adresse.select(["adresse_id", "poi_id"]), on="poi_id", how="left")

    return df_adresse, df


def build_poi_table(df: pl.DataFrame):
    """
    Construit la table POI avec poi_id + adresse_id.
    """

    desired_cols = [
        "poi_id",
        "nom_du_poi",
        "latitude",
        "longitude",
        "main_category_id",
        "sub_category_id",
        "adresse_id",        # <-- la clé étrangère vers adresses
        "final_score",
        "density_commune",
        "diversity_commune",
        "proximity_commune",
        "proximity_region",
        "category_weight",
        # "embedding",
    ]

    existing_cols = [c for c in desired_cols if c in df.columns]

    df_poi = df.select(existing_cols)

    return df_poi


def split_into_tables(df: pl.DataFrame):
    """
    Orchestration complète.
    """

    # 1) Catégories
    df_main_category, df_sub_category, df = build_category_tables(df)

    # 2) POI id
    df = build_poi_id(df)

    # 3) Adresses (avec adresse_id + injection dans df)
    df_adresse, df = build_adresse_table(df)

    # 4) Table POI (avec adresse_id)
    df_poi = build_poi_table(df)

    return {
        "poi": df_poi,
        "adresse": df_adresse,
        "main_category": df_main_category,
        "sub_category": df_sub_category,
    }