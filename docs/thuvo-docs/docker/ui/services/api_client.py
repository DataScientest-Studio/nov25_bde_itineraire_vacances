# docker/ui/services/api_client.py
import os
from typing import Any, Dict, Optional
import requests
import os
import pandas as pd


DEFAULT_API_BASE_URL = "http://localhost:8000"
PRIME_ENDPOINT = "/prime/itinerary"


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def call_prime(payload: Dict[str, Any], *, timeout: int = 30) -> Dict[str, Any]:
    base = get_api_base_url()
    url = f"{base}{PRIME_ENDPOINT}"

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"API request failed: {e}") from e

    if resp.status_code != 200:
        # garde le texte brut pour debug (souvent utile en jury)
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"API returned non-JSON response: {resp.text[:500]}") from e
    



API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

def _get_json(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()

def call_prime(payload: dict):
    r = requests.post(f"{API_BASE_URL}/prime", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def fetch_dt_ctx(limit: int | None = None) -> pd.DataFrame:
    data = _get_json("/ctx/dt", params={"limit": limit} if limit else None)
    return pd.DataFrame(data["rows"])

def fetch_ta_ctx(limit: int | None = None) -> pd.DataFrame:
    data = _get_json("/ctx/ta", params={"limit": limit} if limit else None)
    return pd.DataFrame(data["rows"])

def fetch_ab_ctx(limit: int | None = None) -> pd.DataFrame:
    data = _get_json("/ctx/ab", params={"limit": limit} if limit else None)
    return pd.DataFrame(data["rows"])

