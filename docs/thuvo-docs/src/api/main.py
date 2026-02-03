from fastapi import FastAPI

from api.routes.prime import router as prime_router

app = FastAPI(title="Itinéraire Vacances API")
app.include_router(prime_router)
