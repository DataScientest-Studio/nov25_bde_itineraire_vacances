from __future__ import annotations
import uuid
from fastapi import APIRouter, HTTPException
from src.domain.prime import compute_prime, compute_safe_ranking



# --- Quality gate optionnel (ne doit pas bloquer l'API) ---
try:
    from ops.quality.quality_gate import prime_quality_gate, get_quality_mode
    _QUALITY_ENABLED = True
except Exception:
    prime_quality_gate = None
    get_quality_mode = lambda: "DISABLED"
    _QUALITY_ENABLED = False

router = APIRouter()


@router.get("/prime")
def prime_endpoint(zone: str, limit: int = 50):
    run_id = uuid.uuid4().hex

    # 1) Calcul PRIME (MVP: list[dict])
    results = compute_prime(zone=zone, limit=limit)

    # 2) Quality Gate (optionnel)
    quality = {"mode": get_quality_mode(), "status": "SKIPPED"}
    mode_used = "PRIME"

    if _QUALITY_ENABLED and prime_quality_gate is not None:
        try:
            results, quality = prime_quality_gate(results, run_id=run_id)
        except ValueError as e:
            if str(e) == "PRIME_QUALITY_BLOCKED":
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "PRIME results blocked by quality checks (STRICT mode).",
                        "quality_mode": get_quality_mode(),
                        "run_id": run_id,
                    },
                )
            raise

    # 3) Réponse API
    return {
        "run_id": run_id,
        "zone": zone,
        "mode_used": mode_used,
        "quality": quality,
        "results": results[:limit],}


@router.get("/prime_fallback")
def prime_endpoint_fallback(zone: str, limit: int = 50):
    """
    Variante "prod" : en STRICT, si PRIME est bloqué, on renvoie un fallback safe.
    (Endpoint séparé pour éviter d’avoir 2 @router.get("/prime") dans le même fichier)
    """
    run_id = uuid.uuid4().hex

    results = compute_prime(zone=zone, limit=limit)

    try:
        results, quality = prime_quality_gate(results, run_id=run_id)
        mode_used = "PRIME"
    except ValueError as e:
        if str(e) != "PRIME_QUALITY_BLOCKED":
            raise

        # fallback safe
        results = compute_safe_ranking(zone=zone, limit=limit)
        quality = {
            "mode": get_quality_mode(),
            "status": "BLOCKED",
            "action": "FALLBACK",
            "message": "PRIME blocked by quality gate; fallback ranking used.",
            "run_id": run_id,}
        mode_used = "FALLBACK_SAFE"

    return {
        "run_id": run_id,
        "zone": zone,
        "mode_used": mode_used,
        "quality": quality,
        "results": results[:limit],}
