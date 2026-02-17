# src/ops/quality/quality_gate.py
from __future__ import annotations

"""
QUALITY GATE (DB-first)

Objectif:
- Dans ton projet, les datasets sont chargés en Postgres (RAW/SILVER).
- Les checks "source of truth" sont désormais les fonctions SQL dans le schéma ops:
    - ops.quality_check_datatourisme_prime()
    - ops.quality_check_tripadvisor_france()

Donc ce quality_gate ne charge plus de parquet/csv/json.
Il orchestre les checks DB et bloque (STRICT) si un check SQL échoue.

Vars env:
- RUN_ID (optionnel)
- PRIME_QUALITY_MODE: STRICT | RELAXED (default STRICT)
- TRIPADVISOR_QUALITY_MODE: STRICT | RELAXED (default STRICT)

Connexion DB:
- DB_HOST, DB_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import os
from typing import Literal

import psycopg2

from src.ops.logging.logging_config import get_logger

QualityMode = Literal["STRICT", "RELAXED"]


# --------------------------------------------------------------------------------------
# Mode
# --------------------------------------------------------------------------------------
def _normalize_mode(raw: str) -> QualityMode:
    raw = (raw or "").upper().strip()
    return "RELAXED" if raw == "RELAXED" else "STRICT"


def get_prime_quality_mode() -> QualityMode:
    return _normalize_mode(os.getenv("PRIME_QUALITY_MODE", "STRICT"))


def get_tripadvisor_quality_mode() -> QualityMode:
    return _normalize_mode(os.getenv("TRIPADVISOR_QUALITY_MODE", "STRICT"))


# --------------------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------------------
def _dsn_from_env() -> str:
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "prime")
    user = os.getenv("POSTGRES_USER", "prime")
    pwd = os.getenv("POSTGRES_PASSWORD", "prime")
    return f"host={host} port={port} dbname={db} user={user} password={pwd}"


def _run_sql_check(cur, fn_name: str) -> None:
    """
    Exécute une fonction SQL ops.*_check_*().
    Les fonctions SQL lèvent EXCEPTION si KO -> psycopg2.Error ici.
    """
    cur.execute(f"SELECT {fn_name}();")
    # Certaines fonctions retournent void => fetchone() peut être None selon config.
    # On tente un fetchone sans en dépendre.
    try:
        _ = cur.fetchone()
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Entry point called by pipelines/run_etl.py (mod.main())
# --------------------------------------------------------------------------------------
def main(data=None) -> None:  # data kept for compatibility with run_etl.py signature
    print("\n[QUALITY GATE] Running DB checks...")

    run_id = os.getenv("RUN_ID", "local")

    prime_mode = get_prime_quality_mode()
    trip_mode = get_tripadvisor_quality_mode()

    logger = get_logger("pipeline.quality_gate", run_id=run_id)
    logger.info(
        "quality_gate_start",
        extra={
            "event": "quality_gate_start",
            "run_id": run_id,
            "prime_mode": prime_mode,
            "tripadvisor_mode": trip_mode,
        },
    )

    dsn = _dsn_from_env()

    # We will track failures but only block if mode == STRICT
    prime_ok = True
    trip_ok = True
    prime_err = None
    trip_err = None

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:

            # ============================
            # PRIME (DB function)
            # ============================
            try:
                print("[QUALITY GATE][PRIME] calling ops.quality_check_datatourisme_prime() ...")
                _run_sql_check(cur, "ops.quality_check_datatourisme_prime")
                print("[QUALITY GATE][PRIME] OK")
                logger.info(
                    "prime_quality_ok",
                    extra={"event": "prime_quality_ok", "dataset": "prime"},
                )
            except psycopg2.Error as e:
                prime_ok = False
                prime_err = str(e).strip()
                print("[QUALITY GATE][PRIME] FAILED")
                logger.error(
                    "prime_quality_failed",
                    extra={
                        "event": "prime_quality_failed",
                        "dataset": "prime",
                        "error": prime_err,
                    },
                )
                # On rollback la transaction pour pouvoir continuer / exécuter d'autres queries
                conn.rollback()

            # ============================
            # TRIPADVISOR (DB function)
            # ============================
            try:
                print("[QUALITY GATE][TRIPADVISOR] calling ops.quality_check_tripadvisor_france() ...")
                _run_sql_check(cur, "ops.quality_check_tripadvisor_france")
                print("[QUALITY GATE][TRIPADVISOR] OK")
                logger.info(
                    "tripadvisor_quality_ok",
                    extra={"event": "tripadvisor_quality_ok", "dataset": "tripadvisor"},
                )
            except psycopg2.Error as e:
                trip_ok = False
                trip_err = str(e).strip()
                print("[QUALITY GATE][TRIPADVISOR] FAILED")
                logger.error(
                    "tripadvisor_quality_failed",
                    extra={
                        "event": "tripadvisor_quality_failed",
                        "dataset": "tripadvisor",
                        "error": trip_err,
                    },
                )
                conn.rollback()

    # ============================
    # Decision
    # ============================
    prime_block = (prime_mode == "STRICT" and not prime_ok)
    trip_block = (trip_mode == "STRICT" and not trip_ok)

    if prime_block or trip_block:
        msg = "[QUALITY GATE] FAILED — erreurs bloquantes détectées."
        if prime_block:
            msg += f"\n- PRIME: {prime_err or 'unknown error'}"
        if trip_block:
            msg += f"\n- TRIPADVISOR: {trip_err or 'unknown error'}"

        logger.error(
            "quality_gate_blocked",
            extra={
                "event": "quality_gate_blocked",
                "prime_ok": prime_ok,
                "tripadvisor_ok": trip_ok,
                "prime_mode": prime_mode,
                "tripadvisor_mode": trip_mode,
                "prime_error": prime_err,
                "tripadvisor_error": trip_err,
            },
        )
        raise RuntimeError(msg)

    print("[QUALITY GATE] OK — pipeline autorisé.")
    logger.info(
        "quality_gate_passed",
        extra={
            "event": "quality_gate_passed",
            "prime_ok": prime_ok,
            "tripadvisor_ok": trip_ok,
            "prime_mode": prime_mode,
            "tripadvisor_mode": trip_mode,
        },
    )


if __name__ == "__main__":
    main()
