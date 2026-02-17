from fastapi import APIRouter
from schemas.prime import PrimeRequest, PrimeResponse
from services.prime_service import generate_prime_itinerary

router = APIRouter(tags=["Prime"])

@router.post("/itinerary", response_model=PrimeResponse)
def prime_itinerary(payload: PrimeRequest):
    return generate_prime_itinerary(payload)


@router.get("/ctx/ab")
def ctx_ab(limit: int = 5000):
    from data.loaders import load_airbnb_df
    df = load_airbnb_df(limit=limit)
    return df.to_dict(orient="records")


@router.get("/ctx/ta")
def ctx_ta(limit: int = 5000):
    from data.loaders import load_tripadvisor_df
    df = load_tripadvisor_df(limit=limit)
    return df.to_dict(orient="records")
