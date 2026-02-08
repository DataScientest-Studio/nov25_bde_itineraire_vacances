from app.repositories.poi_repository import POIRepository
import h3

class POIService:

    def __init__(self):
        self.repository = POIRepository()

    def radius_to_kring(self, radius_km):
        # Approximation: 1 k-ring ≈ 7.8 km radius
        return max(1, int(radius_km / 7.8))

    def get_filtered_pois(self, db, filters):
        center_h3 = h3.latlng_to_cell(filters.latitude, filters.longitude, 7)

        k = self.radius_to_kring(filters.radius)
        h3_r7 = list(h3.grid_disk(center_h3, k))

        return self.repository.filter_pois(db, filters, h3_r7)



poi_service = POIService()