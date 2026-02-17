# ======================================================================================================================
# NORMALISATION ADMIN : region / department / postal_code ---> docker/ui/data/normalization_admin.py
# (TODO) A FAIRE ausi DANS ETL
# ======================================================================================================================

from infra.imports import pd, re, unicodedata


def _to_str_series(s: pd.Series) -> pd.Series:
    """
    Convertit une Series en type pandas 'string', remplace les NaN par chaîne vide,
    et supprime les espaces en début/fin.
    Objectif : homogénéiser les colonnes texte avant normalisation.
    """
    # Bon helper: évite NaN, espaces, types bizarres
    return s.astype("string").fillna("").str.strip()


def _digits_only(x: str) -> str:
    """
    Extrait la première séquence de chiffres trouvée dans une valeur.
    - Exemple: "75001 Paris" -> "75001"
    - Exemple: "CP: 33000"  -> "33000"
    - Si aucun chiffre n'est trouvé -> chaîne vide.
    """
    # Sécurise le type : on travaille toujours sur une chaîne
    if not isinstance(x, str):
        x = str(x)
    # prendre la première séquence de chiffres consécutifs
    # => si CP "75001-75002" --> prends "75001"
    # => si "FR-75001" --> prends "75001"
    m = re.search(r"(\d+)", x)
    return m.group(1) if m else ""


def _norm_txt(s: str) -> str:
    """
    Normalise une chaîne de caractères pour des comparaisons robustes.
    Étapes :
    - Sécurise l'entrée (None -> "")
    - Minuscule + suppression des espaces superflus
    - Suppression des accents (Unicode NFD)
    - Harmonisation des caractères usuels (tirets, apostrophes)
    - Normalisation des espaces multiples
    Exemples :
    - "Île-de-France" -> "ile de france"
    - "  Côte d’Azur " -> "cote d'azur"
    """
    # Helper "UX robust filtering": accents/casse/espaces
    s = (s or "").strip().lower()

    # Décomposition Unicode pour supprimer les accents
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    # Harmonisation de caractères fréquents
    s = s.replace("-", " ").replace("’", "'")

    # Réduction des espaces multiples
    s = re.sub(r"\s+", " ", s)

    return s


def normalize_admin_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les colonnes administratives (pays, ville, code postal, département, région)
    pour garantir un schéma commun entre plusieurs sources (ex: DataTourisme, TripAdvisor).
    Règles:
    1) Harmoniser certains noms de colonnes (departement -> department, etc.)
    2) Garantir que toutes les colonnes cibles existent (créées vides si absentes)
    3) Nettoyer les valeurs texte (string, NaN -> "", strip)
    4) Forcer postal_code à ne contenir que des chiffres (extraction)
    5) Si department est vide, le déduire depuis les 2 premiers chiffres du code postal
    """
    # On évite de modifier le DataFrame d'origine
    df = df.copy()

    # 1) Harmonisation des noms de colonnes (département)
    # Certaines sources utilisent "departement" (FR) au lieu de "department"
    if "departement" in df.columns and "department" not in df.columns:
        df = df.rename(columns={"departement": "department"})
    if "département" in df.columns and "department" not in df.columns:
        df = df.rename(columns={"département": "department"})

    # 2) Harmonisation des noms de colonnes (région)
    # Variantes courantes rencontrées : "Region" (majuscule) ou "région" (accent)
    if "Region" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"Region": "region"})
    if "région" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"région": "region"})


    # 3) Garantir l'existence des colonnes attendues
    # Si une colonne est absente, on la crée avec une chaîne vide pour standardiser le schéma
    for col in ["country", "city", "postal_code", "department", "region"]:
        if col not in df.columns:
            df[col] = ""

    # 4) Nettoyage / typage texte
    # On force tout en string + NaN -> "" + suppression des espaces
    for col in ["country", "city", "postal_code", "department", "region"]:
        df[col] = _to_str_series(df[col])

    # 5) Normalisation du code postal
    # On garde uniquement la première séquence de chiffres trouvée
    # CP FR: garder 5 chiffres max pour éviter "750011234"
    df["postal_code"] = df["postal_code"].apply(_digits_only).str[:5]

    # 6) Fallback: département depuis le code postal
    # Si department est vide, on prend les 2 premiers chiffres du CP (convention FR)
    df.loc[df["department"] == "", "department"] = df["postal_code"].str[:2]

    return df


def normalize_all_admin(df_dt, df_ta, df_ab):
    """
    Applique la normalisation admin sur 3 df et renvoie le triplet.
    """
    return (
        normalize_admin_cols(df_dt),
        normalize_admin_cols(df_ta),
        normalize_admin_cols(df_ab),)


def list_regions(df: pd.DataFrame) -> list[str]:
    # Si "region" n'existe pas (source inattendue), éviter KeyError:
    if "region" not in df.columns:
        return []
    vals = df["region"].dropna().astype(str).str.strip()
    # Simple + cache 1h : parfait pour alimenter un selectbox
    return sorted([x for x in vals.unique().tolist() if x])


def list_departments(df: pd.DataFrame, region: str | None) -> list[str]:
    if "department" not in df.columns:
        return []
    d = df
    # Filtre région robuste (accents/casse/espaces) si region est fourni
    if region and "region" in d.columns:
        region_norm = _norm_txt(region)
        d = d[_to_str_series(d["region"]).apply(_norm_txt) == region_norm]
    # Valeurs uniques + tri + supprime vides
    deps = _to_str_series(d["department"]).tolist()
    deps = [x for x in deps if x]
    return sorted(set(deps))


# --- Aliases publics (API stable) ---
norm_txt = _norm_txt
to_str_series = _to_str_series