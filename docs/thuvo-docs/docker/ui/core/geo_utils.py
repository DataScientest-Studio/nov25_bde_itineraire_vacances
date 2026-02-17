# ======================================================================================================================
# PRIME ENGINE — utilitaires ---> docker/ui/core/geo_utils.py
# ======================================================================================================================

from infra.imports import math, pd


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Calcule la distance "à vol d’oiseau" (grand cercle) entre deux points GPS.
    Paramètres
    ----------
    lat1, lon1 : float
        Latitude/longitude du point A (en degrés).
    lat2, lon2 : float
        Latitude/longitude du point B (en degrés).
    Retour
    ------
    float
        Distance en kilomètres entre A et B, via la formule de Haversine.
    """
    # (RESP) Pure fonction moteur : aucune dépendance UI -> facile à extraire dans un module "engine_utils.py"
    # Rayon moyen de la Terre en km
    R = 6371.0

    # Conversion degrés -> radians (la trigonométrie utilise les radians)
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Formule de Haversine
    a = (math.sin(dlat / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    
    # 2*R*asin(sqrt(a)) = distance sur la sphère
    return 2 * R * math.asin(math.sqrt(a))



def safe_haversine_km(lat1, lon1, lat2, lon2):
    """
    Version "safe" de haversine_km : renvoie None si les coordonnées sont invalides.
    Utilité :
    - Évite les crashs quand lat/lon valent None, NaN, ou des strings non convertibles.
    - Pratique pour l'appliquer sur des colonnes de DataFrame.
    Retour
    ------
    float | None
        Distance en km si calculable, sinon None.
    """
    # (RESP) Helper moteur robuste : très utile dans apply DataFrame
    # Gère None/NaN/strings
    try:
        # Vérifications de valeurs manquantes (None ou NaN)
        if (
            lat1 is None or lon1 is None or
            lat2 is None or lon2 is None or
            pd.isna(lat1) or pd.isna(lon1) or
            pd.isna(lat2) or pd.isna(lon2)):
            return None
        # Conversion explicite en float avant calcul
        return haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
    except Exception:
        # En cas d'entrée "sale" (ex: 'abc'), on renvoie None plutôt que planter
        return None
    


def is_strict_radius(max_km: float) -> bool:
    """
    Détermine si on est en mode "rayon strict" (petit rayon).
    Règle : <= 10 km : on considère que l’utilisateur veut rester proche du point d’ancrage.
    Retour
    ------
    bool
    """
    # (RESP) Règle métier / UX: si petit rayon, on considère la contrainte forte
    return float(max_km) <= 3.0



def assign_rings(distance_km: pd.Series, max_km: float) -> pd.Series:
    """
    Attribue un "ANNEAU" de distance à chaque POI (Proche / Intermédiaire / Éloigné).
    Objectif : Donner une lecture simple du niveau d'éloignement relatif à max_km.
    Seuils (adaptatifs)
    -------------------
    a = max(0.5, 0.33 * max_km)
    b = max(1.0, 0.66 * max_km)
    - distance <= a : "Proche"
    - distance <= b : "Intermédiaire"
    - sinon         : "Éloigné"
    Paramètres
    ----------
    distance_km : pd.Series
        Série des distances (km) pour les POIs.
    max_km : float
        Rayon maximum choisi par l'utilisateur.
    Retour
    ------
    pd.Series
        Labels d'anneau pour chaque distance.
    """
    # (RESP) Helper moteur métier: labeling lisible pour l'utilisateur
    # Seuils dynamiques basés sur max_km, avec un minimum pour éviter 0
    max_km = float(max_km)
    a = max(0.5, 0.33 * max_km)
    b = max(1.0, 0.66 * max_km)

    def ring(x):
        if x <= a:
            return "Proche"
        if x <= b:
            return "Intermédiaire"
        return "Éloigné"

    # (NOTE) Suppose distance_km convertible en float (sinon ValueError)
    return distance_km.apply(lambda x: ring(float(x)))



