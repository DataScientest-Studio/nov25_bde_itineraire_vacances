from schemas.prime import PrimeRequest, PrimeResponse, PrimeItem

def generate_prime_itinerary(payload: PrimeRequest) -> PrimeResponse:

    return PrimeResponse(
        anchor_lat=48.8566,
        anchor_lon=2.3522,
        itinerary=[
            PrimeItem(
                jour=1,
                slot="09:00–12:00",
                type="POI Central",
                name="POI exemple (stub API)",
                category=(payload.categories[0] if payload.categories else None),
                score=8.5,
                lat=48.8606,
                lon=2.3376,
            )
        ],
        warnings=[],
    )
