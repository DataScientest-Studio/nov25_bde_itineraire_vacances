# ops/data_checks/prime_checks.py
"""
Contrôles qualité critiques pour PRIME (Gold) - contrat strict + geom.
- Compatible list[dict] ou DataFrame
- ERROR = bloquant / WARN = alerte
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


Row = Mapping[str, Any]
Rows = list[dict[str, Any]]
InputLike = "pd.DataFrame | Rows"


@dataclass(frozen=True)
class Issue:
    severity: str
    check: str
    message: str
    n_rows: int
    sample_ids: list[Any]


REQUIRED_COLS: list[str] = [
    "source_id",
    "lat",
    "lon",
    "geom",
    "city",
    "region",
    "rating",
    "review_value_label_fr",
    "locomotion_mode_fr",
    "is_label_incontournable",
    "price_level",
    "main_category",
    "sub_category",
    "type_principal",
    "main_cat_weight",
    "format_weight",
    "tempo_weight",
    "score_prime",
    "hebergement_type",
    "is_active",
]

NON_EMPTY_TEXT_COLS: list[str] = ["city", "region", "main_category", "type_principal"]

RANGE_RULES: dict[str, tuple[float, float]] = {
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
    "rating": (0.0, 5.0),
    "main_cat_weight": (0.0, 10.0),
    "format_weight": (-1.0, 1.0),
    "tempo_weight": (-1.0, 1.0),
    "score_prime": (0.0, 10_000.0),
}

PRICE_LEVEL_ALLOWED = {"eco", "normal", "confort", "premium"}


def _is_rows(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))


def _iter_rows(data: InputLike) -> Iterable[dict[str, Any]]:
    if _is_rows(data):
        return data  # type: ignore[return-value]
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        rows = to_dict(orient="records")
        if not isinstance(rows, list):
            raise TypeError("to_dict(orient='records') must return list[dict].")
        return rows  # type: ignore[return-value]
    raise TypeError("Unsupported input type (expected list[dict] or DataFrame-like).")


def _get_columns(data: InputLike) -> set[str]:
    if _is_rows(data):
        cols: set[str] = set()
        for row in data:
            cols |= set(row.keys())
        return cols
    cols = getattr(data, "columns", None)
    if cols is None:
        raise TypeError("Unsupported input type (expected DataFrame-like with .columns).")
    return set(cols)


def _as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _sample_ids(rows: Sequence[Row], mask: Sequence[bool], k: int = 10) -> list[Any]:
    out: list[Any] = []
    for i, ok in enumerate(mask):
        if ok:
            out.append(rows[i].get("source_id", i))
            if len(out) >= k:
                break
    return out


def check_required_schema(data: InputLike) -> list[Issue]:
    cols = _get_columns(data)
    rows = list(_iter_rows(data))
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        return [
            Issue("ERROR", "required_schema", f"Colonnes manquantes : {missing}", len(rows), [])
        ]

    bad_sid = [False] * len(rows)
    bad_geom = [False] * len(rows)

    for i, r in enumerate(rows):
        bad_sid[i] = _is_blank(r.get("source_id"))
        bad_geom[i] = r.get("geom") is None  # en gold MV, on filtre geom IS NOT NULL

    issues: list[Issue] = []
    if any(bad_sid):
        issues.append(Issue("ERROR", "required_schema", "source_id vide/null", sum(bad_sid), _sample_ids(rows, bad_sid)))
    if any(bad_geom):
        issues.append(Issue("ERROR", "required_schema", "geom NULL", sum(bad_geom), _sample_ids(rows, bad_geom)))

    return issues


def check_unique_source_id(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    seen: set[Any] = set()
    dup = [False] * len(rows)
    for i, r in enumerate(rows):
        sid = r.get("source_id")
        if sid in seen:
            dup[i] = True
        else:
            seen.add(sid)
    if any(dup):
        return [Issue("ERROR", "unique_source_id", "source_id dupliqués", sum(dup), _sample_ids(rows, dup))]
    return []


def check_ranges_and_types(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    issues: list[Issue] = []

    for col, (lo, hi) in RANGE_RULES.items():
        bad_type = [False] * len(rows)
        out = [False] * len(rows)

        for i, r in enumerate(rows):
            v_raw = r.get(col)
            v = _as_float(v_raw)
            if v is None and v_raw is not None:
                bad_type[i] = True
                continue
            if v is None:
                continue
            if v < lo or v > hi:
                out[i] = True

        if any(bad_type):
            issues.append(Issue("ERROR", "ranges_and_types", f"{col} non numérique", sum(bad_type), _sample_ids(rows, bad_type)))

        if any(out):
            sev = "ERROR" if col in {"lat", "lon"} else "WARN"
            issues.append(Issue(sev, "ranges_and_types", f"{col} hors bornes [{lo},{hi}]", sum(out), _sample_ids(rows, out)))

    return issues


def check_non_empty_text(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    issues: list[Issue] = []
    for col in NON_EMPTY_TEXT_COLS:
        bad = [False] * len(rows)
        for i, r in enumerate(rows):
            bad[i] = _is_blank(r.get(col))
        if any(bad):
            issues.append(Issue("WARN", "non_empty_text", f"Valeurs vides dans {col}", sum(bad), _sample_ids(rows, bad)))
    return issues


def check_booleans(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    issues: list[Issue] = []
    for col in ("is_label_incontournable", "is_active"):
        bad = [False] * len(rows)
        for i, r in enumerate(rows):
            v = r.get(col)
            if v is None:
                continue
            if not _is_bool(v):
                bad[i] = True
        if any(bad):
            issues.append(Issue("WARN", "booleans", f"{col} n'est pas bool (True/False/NULL)", sum(bad), _sample_ids(rows, bad)))
    return issues


def check_price_level(data: InputLike) -> list[Issue]:
    rows = list(_iter_rows(data))
    bad = [False] * len(rows)
    for i, r in enumerate(rows):
        pl = r.get("price_level")
        if _is_blank(pl):
            continue
        if not isinstance(pl, str):
            bad[i] = True
            continue
        if pl.strip().lower() not in PRICE_LEVEL_ALLOWED:
            bad[i] = True
    if any(bad):
        return [Issue("WARN", "price_level", "price_level inattendu (eco|normal|confort|premium)", sum(bad), _sample_ids(rows, bad))]
    return []


def check_score_prime_formula(data: InputLike, tol: float = 1e-6) -> list[Issue]:
    rows = list(_iter_rows(data))
    bad = [False] * len(rows)
    for i, r in enumerate(rows):
        mcw = _as_float(r.get("main_cat_weight"))
        fw = _as_float(r.get("format_weight"))
        tw = _as_float(r.get("tempo_weight"))
        sp = _as_float(r.get("score_prime"))
        if None in (mcw, fw, tw, sp):
            continue
        expected = mcw * (1.0 + fw + tw)
        if abs(expected - sp) > tol:
            bad[i] = True
    if any(bad):
        return [Issue("ERROR", "score_prime_formula", "score_prime incohérent avec la formule PRIME", sum(bad), _sample_ids(rows, bad))]
    return []


def run_prime_checks(data: InputLike) -> list[Issue]:
    issues = check_required_schema(data)
    if any(i.severity == "ERROR" and i.check == "required_schema" for i in issues):
        return issues

    issues += check_unique_source_id(data)
    issues += check_ranges_and_types(data)
    issues += check_non_empty_text(data)
    issues += check_booleans(data)
    issues += check_price_level(data)
    issues += check_score_prime_formula(data)
    return issues
