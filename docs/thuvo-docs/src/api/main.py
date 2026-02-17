# src/api/main.py
import os
from fastapi import FastAPI
from sqlalchemy import create_engine, text

from .routes.prime import router as prime_router
from .routes.ctx import router as ctx_router


# --------------------------------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------------------------------
app = FastAPI(
    title="Itineraire Vacances API",
    version="0.1.0",
)


# --------------------------------------------------------------------------------------------------
# DATABASE (optionnel au démarrage)
# --------------------------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# --------------------------------------------------------------------------------------------------
# ROUTERS
# --------------------------------------------------------------------------------------------------
app.include_router(prime_router)
app.include_router(ctx_router)


# --------------------------------------------------------------------------------------------------
# HEALTHCHECK
# --------------------------------------------------------------------------------------------------
@app.get("/health")
def health():
    if engine is None:
        return {"status": "degraded", "db": "missing DATABASE_URL"}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        return {"status": "degraded", "db": "error", "detail": str(e)}


# simple ping (debug réseau)
@app.get("/ping")
def ping():
    return {"ok": True}
