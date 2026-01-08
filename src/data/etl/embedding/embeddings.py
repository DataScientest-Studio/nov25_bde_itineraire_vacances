# embeddings.py

import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer


# Charger le modèle une seule fois
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 1) Construire une colonne texte riche pour les embeddings
# ---------------------------------------------------------
def build_text_embedding_column(
    df: pl.DataFrame,
    columns: list[str] = None,
    separator: str = " | "
) -> pl.DataFrame:
    """
    Construit une colonne 'text_embedding' en concaténant plusieurs colonnes
    (name, description, main_category, sub_category, etc.)
    en gérant automatiquement les valeurs None.

    columns : liste des colonnes à concaténer
              si None → utilise un set par défaut
    """

    if columns is None:
        columns = [
            "nom_du_poi",
            "description",
            "type_principal"
            "main_category",
            "sub_category",
        ]

    # Remplacer les colonnes manquantes par ""
    exprs = []
    for col in columns:
        if col in df.columns:
            exprs.append(pl.col(col).fill_null(""))
        else:
            exprs.append(pl.lit(""))

        exprs.append(pl.lit(separator))

    # Retirer le dernier séparateur
    exprs = exprs[:-1]

    return df.with_columns(
        pl.concat_str(exprs).alias("text_embedding")
    )


# ---------------------------------------------------------
# 2) Ajouter les embeddings SentenceTransformer
# ---------------------------------------------------------
def add_embeddings(
    df: pl.DataFrame,
    text_column: str = "text_embedding",
    batch_size: int = 64,
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Ajoute une colonne 'embedding' contenant un vecteur (list[float])
    pour chaque POI, basé sur SentenceTransformer.

    - Encodage batch pour performance
    - Gestion des valeurs None
    - Embeddings normalisés (optionnel)
    """

    # 1) Extraire les textes
    texts = df[text_column].to_list()
    texts = [t if t is not None else "" for t in texts]

    # 2) Encoder en batch
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    # 3) Convertir en listes Python
    embeddings_list = embeddings.tolist()

    # 4) Ajouter la colonne dans Polars
    return df.with_columns(
        pl.Series("embedding", embeddings_list)
    )