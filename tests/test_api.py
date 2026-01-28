import requests

payload = {
  "commune": "Paris",
  "main_categories": ["Patrimoine & Monuments", "Gastronomie & Restauration", "Shopping & Artisanat"],
  "sub_categories" : ["Restaurants","Bibliothèques & médiation","Restauration rapide","Bars & cafés","Religieux"],
  "min_score": 0.15,
  "nb_days": 3,
  "start": {"lat": 48.86666, "lon": 2.33333},
  "osrm_mode": "walk",
  "solver": "auto"

}

r = requests.post("http://localhost:8000/itinerary", json=payload)

print("Status:", r.status_code)
print("Response:", r.json())