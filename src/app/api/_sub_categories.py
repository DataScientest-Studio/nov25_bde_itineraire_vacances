import requests

def fetch_sub_categories(main_cat):
    url = "http://localhost:8000/sub_categories"
    payload = {"main_categories": [main_cat]}  # conforme à ton modèle Pydantic

    try:
        print("➡️ Envoi du payload :", payload)
        response = requests.post(url, json=payload)
        print("➡️ Status code :", response.status_code)

        response.raise_for_status()

        data = response.json()
        print("➡️ Réponse JSON :", data)

        return data["sub_categories"]

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de la récupération des données : {e}")


if __name__ == "__main__":
    # Exemple : remplace par une vraie catégorie existante
    main_category = "Gastronomie & Restauration"
    subcats = fetch_sub_categories(main_category)

    print("\nSous-catégories récupérées :")
    print(subcats)