from __future__ import annotations
import os
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Tuple, Union
from ops.data_checks.prime_checks import Issue, InputLike, run_prime_checks
from ops.logging.logging_config import get_logger

"""
Quality Gate : applique les checks PRIME, logge un résumé, et décide de bloquer ou non.
- STRICT  : bloque si n_errors > 0
- RELAXED : ne bloque jamais, mais renvoie les infos qualité
"""

QualityMode = Literal["STRICT", "RELAXED"]


def get_quality_mode() -> QualityMode:
    mode = os.getenv("PRIME_QUALITY_MODE", "STRICT").upper().strip()
    return "RELAXED" if mode == "RELAXED" else "STRICT"


def build_quality_payload(issues: List[Issue]) -> Dict[str, Any]:
    """
    Payload léger à renvoyer au client (UI) + à logger.
    """
    n_errors = sum(1 for i in issues if i.severity == "ERROR")
    n_warns = sum(1 for i in issues if i.severity == "WARN")

    by_check: Dict[str, int] = {}
    for i in issues:
        by_check[i.check] = by_check.get(i.check, 0) + 1

    # On limite le détail renvoyé (sinon payload trop gros)
    top_issues = [asdict(i) for i in issues[:10]]

    return {
        "n_issues": len(issues),
        "n_errors": n_errors,
        "n_warns": n_warns,
        "by_check": by_check,
        "top_issues": top_issues,}


def prime_quality_gate(
    data_prime: InputLike,
    *,
    run_id: str,
    mode: QualityMode | None = None,
) -> Tuple[InputLike, Dict[str, Any]]:
    """
    Exécute les checks PRIME et renvoie (data, quality_payload).
    - data peut être une list[dict] (MVP) ou un DataFrame (si tu reviens à pandas plus tard)
    - En mode STRICT, peut lever ValueError("PRIME_QUALITY_BLOCKED")
    """
    logger = get_logger("api.prime.quality", run_id=run_id)
    mode = mode or get_quality_mode()

    issues = run_prime_checks(data_prime)
    payload = build_quality_payload(issues)
    payload["mode"] = mode

    # Log résumé (monitoring)
    logger.info(
        "prime_quality",
        extra={
            "event": "prime_quality",
            "mode": mode,
            "n_issues": payload["n_issues"],
            "n_errors": payload["n_errors"],
            "n_warns": payload["n_warns"],
            "by_check": payload["by_check"],},)

    # Mode STRICT : on bloque si erreurs
    if mode == "STRICT" and payload["n_errors"] > 0:
        logger.error(
            "prime_quality_block",
            extra={
                "event": "prime_quality_block",
                "mode": mode,
                "n_errors": payload["n_errors"],
                "top_issues": payload["top_issues"],},)
        raise ValueError("PRIME_QUALITY_BLOCKED")

    # Mode RELAXED : on laisse passer (mais on garde le payload)
    return data_prime, payload
