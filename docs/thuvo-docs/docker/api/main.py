import os
from fastapi import FastAPI
from sqlalchemy import create_engine, text

from routes.prime import router as prime_router
from routes.ctx import router as ctx_router

# 1) créer l'app
app = FastAPI(title="ItineraireVacances API")

# 2) brancher les routers
app.include_router(prime_router)
app.include_router(ctx_router)

# 3) config DB
DATABASE_URL = os.environ.get("DATABASE_URL", "")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        return {"status": "db_down"}
