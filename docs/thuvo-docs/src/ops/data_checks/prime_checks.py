# ops/data_checks/prime_checks.py
"""
Contrôles qualité critiques pour les résultats PRIME (Gold).

Objectifs :
- Vérifier la cohérence des données AVANT exposition UI/API
- Séparer les erreurs bloquantes (ERROR) des alertes (WARN)
- Fonctionner SANS pandas : compatible list[dict] (MVP) et DataFrame si présent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------
# Types d'entrée supportés
# ---------------------------------------------------------------------
Row = Mapping[str, Any]
Rows = list[dict[str, Any]]
InputLike = "pd.DataFrame | Rows"  # pandas optionnel (string annotation)


# ---------------------------------------------------------------------
# Modèle d'issue
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Issue:
    severity: str          # "ERROR" ou "WARN"
    check: str             # nom du contrôle
    message: str           # description lisible
    n_rows: int            # nombre de lignes impactées
    sample_ids: list[Any]  # exemples de poi_id (ou index)


# ---------------------------------------------------------------------
# Règles
# ---------------------------------------------------------------------
REQUIRED_COLS: list[str] = [
    "poi_id",
    "main_cat_weight",
    "format_weight",
    "tempo_weight",
    "final_score",
    "lat",
    "lon",
    "is_active",
]

RANGE_RULES: dict[str, tuple[float, float]] = {
    "main_cat_weight": (0.0, 10.0),
    "format_weight": (-1.0, 1.0),
    "tempo_weight": (-1.0, 1.0),
    "final_score": (0.0, 100.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
}


# ---------------------------------------------------------------------
# Helpers "sans pandas"
# ---------------------------------------------------------------------
def _is_rows(obj: Any) -> bool:
    """True si obj ressemble à une list[dict]."""
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))


def _iter_rows(data: InputLike) -> Iterable[dict[str, Any]]:
    """
    Itère sur les lignes sous forme dict :
    - list[dict] : direct
    - DataFrame : via to_dict(orient="records") si dispo
    """
    if _is_rows(data):
        # on garantit le type dict[str, Any]
        return data  # type: ignore[return-value]

    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        rows = to_dict(orient="records")
        if not isinstance(rows, list):
            raise TypeError("DataFrame.to_dict(orient='records') must return a list of dict.")
        return rows  # type: ignore[return-value]

    raise TypeError(
        "Unsupported PRIME input type (expected list[dict] or DataFrame-like with to_dict(orient='records'))."
    )


def _get_columns(data: InputLike) -> set[str]:
    """
    Retourne l'ensemble des colonnes disponibles :
    - list[dict] : union des keys
    - DataFrame-like : attribute .columns
    """
    if _is_rows(data):
        cols: set[str] = set()
        for row in data:
            cols |= set(row.keys())
        return cols

    cols = getattr(data, "columns", None)
    if cols is None:
        raise TypeError("Unsupported PRIME input type (expected DataFrame-like with .columns).")
    return set(cols)


def _as_float(x: Any) -> float | None:
    """Convertit x en float si possible, sinon None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _sample_ids_from_rows(rows: Sequence[Row], mask: Sequence[bool], k: int = 10) -> list[Any]:
    """
    Extrait quelques poi_id (ou index) pour debug, sans exposer tout le dataset.
    """
    sample: list[Any] = []
    n = min(len(rows), len(mask))
    for i in range(n):
        if mask[i]:
            poi_id = rows[i].get("poi_id", i) if isinstance(rows[i], Mapping) else i
            sample.append(poi_id)
            if len(sample) >= k:
                break
    return sample


# ---------------------------------------------------------------------
# Check 1 : Schéma requis + poi_id non nul
# ---------------------------------------------------------------------
def check_required_schema(data: InputLike) -> list[Issue]:
    cols = _get_columns(data)
    missing = [c for c in REQUIRED_COLS if c not in cols]

    rows = list(_iter_rows(data))
    issues: list[Issue] = []

    if missing:
        issues.append(
            Issue(
                severity="ERROR",
                check="required_schema",
                message=f"Colonnes manquantes : {missing}",
                n_rows=len(rows),
                sample_ids=[],
            )
        )
        return issues  # bloquant

    bad_mask = []
    for row in rows:
        poi_id = row.get("poi_id")
        bad_mask.append(poi_id is None or (isinstance(poi_id, str) and poi_id.strip() == ""))

    if any(bad_mask):
        issues.append(
            Issue(
                severity="ERROR",
                check="required_schema",
                message="poi_id contient des valeurs nulles/vides",
                n_rows=sum(bad_mask),
                sample_ids=_sample_ids_from_rows(rows, bad_mask),
            )
        )

    return issues


# ---------------------------------------------------------------------
# Check 2 : Unicité poi_id
# ---------------------------------------------------------------------
def check_unique_poi_id(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    seen: set[Any] = set()
    dup_mask = [False] * len(rows)

    for i, row in enumerate(rows):
        poi_id = row.get("poi_id")
        if poi_id in seen:
            dup_mask[i] = True
        else:
            seen.add(poi_id)

    if any(dup_mask):
        return [
            Issue(
                severity="ERROR",
                check="unique_poi_id",
                message="poi_id dupliqués (déduplication attendue en Silver/Gold)",
                n_rows=sum(dup_mask),
                sample_ids=_sample_ids_from_rows(rows, dup_mask),
            )
        ]

    return []


# ---------------------------------------------------------------------
# Check 3 : Types numériques + bornes
# ---------------------------------------------------------------------
def check_ranges_and_types(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    issues: list[Issue] = []

    for col, (lo, hi) in RANGE_RULES.items():
        bad_type_mask = [False] * len(rows)
        out_mask = [False] * len(rows)

        for i, row in enumerate(rows):
            v_raw = row.get(col)
            v = _as_float(v_raw)

            # valeur présente mais non convertible => ERROR
            if v is None and v_raw is not None:
                bad_type_mask[i] = True
                continue

            # valeur absente => on ignore (tu peux durcir si besoin)
            if v is None:
                continue

            if v < lo or v > hi:
                out_mask[i] = True

        if any(bad_type_mask):
            issues.append(
                Issue(
                    severity="ERROR",
                    check="ranges_and_types",
                    message=f"Valeurs non numériques dans {col}",
                    n_rows=sum(bad_type_mask),
                    sample_ids=_sample_ids_from_rows(rows, bad_type_mask),
                )
            )

        if any(out_mask):
            issues.append(
                Issue(
                    severity="WARN",
                    check="ranges_and_types",
                    message=f"{col} hors bornes [{lo}, {hi}]",
                    n_rows=sum(out_mask),
                    sample_ids=_sample_ids_from_rows(rows, out_mask),
                )
            )

    return issues


# ---------------------------------------------------------------------
# Check 4 : Formule PRIME
# final_score = main_cat_weight * (1 + format_weight + tempo_weight)
# ---------------------------------------------------------------------
def check_final_score_formula(data: InputLike, tol: float = 1e-6) -> list[Issue]:
    rows = list(_iter_rows(data))
    bad_mask = [False] * len(rows)

    for i, row in enumerate(rows):
        mcw = _as_float(row.get("main_cat_weight"))
        fw = _as_float(row.get("format_weight"))
        tw = _as_float(row.get("tempo_weight"))
        fs = _as_float(row.get("final_score"))

        # si une valeur est manquante/non numérique, on ne double pas l'erreur (déjà check_ranges)
        if None in (mcw, fw, tw, fs):
            continue

        expected = mcw * (1.0 + fw + tw)
        if abs(expected - fs) > tol:
            bad_mask[i] = True

    if any(bad_mask):
        return [
            Issue(
                severity="ERROR",
                check="final_score_formula",
                message="final_score incohérent avec la formule PRIME",
                n_rows=sum(bad_mask),
                sample_ids=_sample_ids_from_rows(rows, bad_mask),
            )
        ]

    return []


# ---------------------------------------------------------------------
# Check 5 : Distribution outliers (robuste, sans numpy)
# Méthode MAD : outlier si |z_robust| > z_thresh
# ---------------------------------------------------------------------
def check_distribution_anomalies(data: InputLike, z_thresh: float = 6.0, min_n: int = 30) -> list[Issue]:
    rows = list(_iter_rows(data))

    scores: list[float] = []
    score_idx: list[int] = []
    for i, row in enumerate(rows):
        fs = _as_float(row.get("final_score"))
        if fs is not None:
            scores.append(fs)
            score_idx.append(i)

    if len(scores) < min_n:
        return []

    # median
    s_sorted = sorted(scores)
    mid = len(s_sorted) // 2
    median = s_sorted[mid] if len(s_sorted) % 2 == 1 else (s_sorted[mid - 1] + s_sorted[mid]) / 2.0

    # MAD
    abs_dev = [abs(x - median) for x in scores]
    abs_dev_sorted = sorted(abs_dev)
    mid2 = len(abs_dev_sorted) // 2
    mad = abs_dev_sorted[mid2] if len(abs_dev_sorted) % 2 == 1 else (abs_dev_sorted[mid2 - 1] + abs_dev_sorted[mid2]) / 2.0

    if mad == 0:
        return []

    out_mask = [False] * len(rows)
    for i_row, x in zip(score_idx, scores):
        robust_z = 0.6745 * (x - median) / mad
        if abs(robust_z) > z_thresh:
            out_mask[i_row] = True

    if any(out_mask):
        return [
            Issue(
                severity="WARN",
                check="distribution_anomalies",
                message="Outliers détectés dans la distribution des scores PRIME",
                n_rows=sum(out_mask),
                sample_ids=_sample_ids_from_rows(rows, out_mask),
            )
        ]

    return []


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_prime_checks(data: InputLike) -> list[Issue]:
    """
    Point d’entrée unique pour exécuter tous les contrôles PRIME.
    Compatible list[dict] et DataFrame.
    """
    issues = check_required_schema(data)

    # schéma incomplet = bloquant : on s'arrête pour éviter du bruit
    if any(i.severity == "ERROR" and i.check == "required_schema" for i in issues):
        return issues

    issues += check_unique_poi_id(data)
    issues += check_ranges_and_types(data)
    issues += check_final_score_formula(data)
    issues += check_distribution_anomalies(data)
    return issues
