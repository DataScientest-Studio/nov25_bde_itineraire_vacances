# Notebooks

## 1. Notebooks pour extraire et visualiser des données
Ce dossier contient les notebooks utilisés pour l'extraction et le calcul des métriques.

### Ordre d'exécution
`1.0-gep-data-extraction-osm.ipynb`  
   - Récupère les données brutes depuis la source externe OSM (OpenStreetMap).  
   - Génère un fichier JSON dans `data/raw/` avec un timestamp "%Y%m%d_%H%M%S".


`2.0-gep-data-exploration.ipynb`  
   - Lit le fichier généré par `1.0-gep-data-extraction-osm.ipynb`.  
   - Calcule les métriques (pourcentage de données manquantes, histogrammes, etc.).  
   - Produit des visualisations interactives avec Plotly.

`3.0-gep-data-exploration_datatourisme.ipynb`  
   - Lit le fichier parquet généré par `src\data\main.py`.
   - Calcule les métriques (pourcentage de données manquantes, histogrammes, etc.).  
   - Produit des visualisations interactives avec Plotly.   

### Données
- Les fichiers volumineux sont stockés dans `data/raw/` (non versionnés, voir `.gitignore`).  
```json
[
  {
    "osm_id":"60949374",
    "category":"monuments",
    "name":"Sans nom",
    "lat":48.6383557,
    "lon":2.2392297,
    "address":"",
    "city":"",
    "phone":"",
    "website":"",
    "opening_hours":"",
    "wikipedia":"",
    "timestamp":"2023-03-12T21:44:30Z"
  },
  {
    "osm_id":"91533012",
    "category":"monuments",
    "name":"Ob\u00e9lisque de la Reine",
    "lat":48.3613074,
    "lon":2.8653626,
    "address":"",
    "city":"",
    "phone":"",
    "website":"",
    "opening_hours":"",
    "wikipedia":"",
    "timestamp":"2025-10-12T13:42:54Z"
  }
]


## 2. Notebooks pour tester des API
Ils se trouvent dans le sous dossier notebooks/api_test:

- `gep-api_routage_comparaison.ipynb` sert à comparer 2 APIs de routage comme osr et osrm.
- `gep-api_tomtom_traffic.ipynb` sert à tester l'API de TOMTOM traffic afin d'en extraire le traffic en temps réel d'une route.


## Environnement
### Installation avec pip

1. Créez un environnement virtuel :
   ```bash
   python -m venv venv

2. Activez l'environnement :
   ```bash
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate # Windows

3. Installez les packages :
   ```bash
   pip install -r requirements.txt

4. relier avec jupyter :
   ```bash
   python -m ipykernel install --user --name=venv --display-name venv

5. Lance Jupyter et choisi le kernel venv

6. Désactivez l'environnement :
   ```bash
   deactivate