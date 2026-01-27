# domain/prime.py
from __future__ import annotations
from typing import Dict, List


def compute_prime(zone: str, limit: int = 50) -> List[Dict]:
    """MVP PRIME: renvoie une liste de résultats (pas de pandas)."""
    results: List[Dict] = []

    for i in range(1, 101):
        main_cat_weight = 1.5
        format_weight = 0.2
        tempo_weight = 0.1
        final_score = main_cat_weight * (1 + format_weight + tempo_weight)

        results.append({
            "poi_id": f"{zone}_poi_{i}",
            "main_cat_weight": main_cat_weight,
            "format_weight": format_weight,
            "tempo_weight": tempo_weight,
            "final_score": final_score,
            "lat": 48.85,
            "lon": 2.35,
            "is_active": True,})

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:limit]


def compute_safe_ranking(zone: str, limit: int = 50) -> List[Dict]:
    """Fallback safe si PRIME est bloqué."""
    return [{
        "poi_id": f"{zone}_safe_{i}",
        "main_cat_weight": 1.0,
        "format_weight": 0.0,
        "tempo_weight": 0.0,
        "final_score": 1.0,
        "lat": 48.85,
        "lon": 2.35,
        "is_active": True,
    } for i in range(1, limit + 1)]
