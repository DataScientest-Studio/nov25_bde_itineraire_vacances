# src/ops/data_checks/tripadvisor_checks.py

# ... (imports identiques)

REQUIRED_COLS: list[str] = [
    "source_id",
    "type",
    "lat",
    "lon",
    "city",
    "rating",
    "review_count",
    "price_level",  # eco|normal|confort|premium
    "reco_score",
    "geom",
    "is_active",
]

RECOMMENDED_COLS: list[str] = [
    "cuisine_continent",
    "vegetarian_friendly",
    "vegan_options",
    "gluten_free",
    "is_halal",
    "is_kosher",
    "ingested_at",
]

# ... RANGE_RULES identiques
# ... PRICE_LEVEL_ALLOWED identique
# ... BOOL_COLS identique mais on ajoute is_active
BOOL_COLS = [
    "vegetarian_friendly",
    "vegan_options",
    "gluten_free",
    "is_halal",
    "is_kosher",
    "is_active",
]

# ---------------------------------------------------------------------
# Check 1 : Schéma requis + id/type non vide + geom non null (bloquant)
# ---------------------------------------------------------------------
def check_required_schema(data: InputLike) -> list[Issue]:
    cols = _get_columns(data)
    rows = list(_iter_rows(data))
    issues: list[Issue] = []

    missing = [c for c in REQUIRED_COLS if c not in cols]
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

    missing_reco = [c for c in RECOMMENDED_COLS if c not in cols]
    if missing_reco:
        issues.append(
            Issue(
                severity="WARN",
                check="recommended_schema",
                message=f"Colonnes recommandées absentes : {missing_reco}",
                n_rows=len(rows),
                sample_ids=[],
            )
        )

    bad_mask = [False] * len(rows)
    bad_geom = [False] * len(rows)

    for i, row in enumerate(rows):
        bad_mask[i] = _is_blank(row.get("source_id")) or _is_blank(row.get("type"))
        bad_geom[i] = row.get("geom") is None  # GOLD filtre déjà geom IS NOT NULL, mais check = garde-fou

    if any(bad_mask):
        issues.append(
            Issue(
                severity="ERROR",
                check="required_schema",
                message="source_id ou type contient des valeurs nulles/vides",
                n_rows=sum(bad_mask),
                sample_ids=_sample_ids_from_rows(rows, bad_mask),
            )
        )

    if any(bad_geom):
        issues.append(
            Issue(
                severity="ERROR",
                check="required_schema",
                message="geom est NULL",
                n_rows=sum(bad_geom),
                sample_ids=_sample_ids_from_rows(rows, bad_geom),
            )
        )

    return issues
