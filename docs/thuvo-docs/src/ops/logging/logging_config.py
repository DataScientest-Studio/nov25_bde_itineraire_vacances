# ops/logging/logging_config.py
"""
Logging centralisé et structuré (JSON) pour l'ensemble du projet.
Objectifs :
- Logs lisibles par un humain (console)
- Logs exploitables par une machine (monitoring, métriques, audit)
- Même format pour ETL, API et scoring PRIME
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """
    Formatter custom pour produire des logs au format JSON.
    Avantages :
    - Standardisable
    - Facilement indexable (ELK, Datadog, etc.)
    - Ajout simple de champs métier via logging.extra
    """

    def format(self, record: logging.LogRecord) -> str:
        # Champs standards communs à tous les logs
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),   # horodatage UTC
            "level": record.levelname,                      # INFO / WARN / ERROR
            "logger": record.name,                          # nom logique du logger
            "msg": record.getMessage(),                     # message principal
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,}

        # Champs métier optionnels passés via logging.extra
        # ex: logger.info("...", extra={"event": "prime_rank", "zone": "Paris-05"})
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "message", "exc_info", "exc_text", "stack_info"):
                continue
            if k.startswith("_"):
                continue
            # On évite de dupliquer les attributs internes de LogRecord
            if k in (
                "name", "levelname", "levelno", "pathname", "filename", "module",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process"):
                continue
            payload[k] = v

        # En cas d’exception, on ajoute la stack trace
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _level_from_env(default: str = "INFO") -> int:
    """
    Détermine le niveau de log à partir de la variable d’environnement LOG_LEVEL.
    Permet de changer le niveau sans modifier le code.
    """
    level_str = os.getenv("LOG_LEVEL", default).upper().strip()
    return getattr(logging, level_str, logging.INFO)


def get_logger(name: str, *, run_id: Optional[str] = None) -> logging.Logger:
    """
    Factory de logger standardisé.

    - Un seul handler (évite les doublons en notebook / reload)
    - Format JSON
    - run_id optionnel pour tracer une exécution complète (ETL, API call, démo)
    """
    logger = logging.getLogger(name)
    logger.setLevel(_level_from_env())

    # Évite d’empiler plusieurs handlers si la fonction est rappelée
    if getattr(logger, "_configured", False):
        if run_id is not None:
            return logging.LoggerAdapter(logger, {"run_id": run_id})  # type: ignore
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_level_from_env())
    handler.setFormatter(JsonFormatter())

    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False

    setattr(logger, "_configured", True)

    # Logger enrichi avec run_id (traçabilité transverse)
    if run_id is not None:
        return logging.LoggerAdapter(logger, {"run_id": run_id})  # type: ignore
    return logger


class Timer:
    """
    Context manager pour mesurer la durée d’un bloc.
    Usage :
        with Timer(logger, event="etl_step", step="extract"):
            ...
    Avantage :
    - Mesures homogènes
    - Zéro boilerplate
    - Logs exploitables pour le monitoring
    """

    def __init__(self, logger: logging.Logger, **fields: Any):
        self.logger = logger
        self.fields = fields
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info(
            "start",
            extra={"event": "timer_start", **self.fields},)
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((time.perf_counter() - self.t0) * 1000)

        # Cas nominal
        if exc is None:
            self.logger.info(
                "end",
                extra={"event": "timer_end", "duration_ms": duration_ms, **self.fields},)
            return False

        # Cas erreur : on log + on laisse remonter l’exception
        self.logger.error(
            "failed",
            extra={
                "event": "timer_end",
                "duration_ms": duration_ms,
                "error_type": str(exc_type),
                "error": str(exc),
                **self.fields,},
            exc_info=True,)
        return False
