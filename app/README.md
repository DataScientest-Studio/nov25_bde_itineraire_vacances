# 🗺️ API Itinéraires de Vacances

Application FastAPI pour le calcul d'itinéraires de vacances optimisés avec clustering spatial et algorithmes de résolution.

## Architecture

```
app/
├── api/                    # Routes FastAPI
│   ├── categories.py       # Gestion des catégories POI
│   ├── poi.py             # Points d'intérêt
│   ├── itinerary.py        # Calcul d'itinéraires
│   └── _api_itinerary_compute.py  # Logique de calcul détaillée
├── core/                   # Configuration système
│   └── database.py        # Gestionnaire de base de données
├── models/                 # Modèles Pydantic
│   ├── categories.py       # Catégories et sous-catégories
│   ├── poi.py             # Structure des POI
│   └── itinerary.py        # Modèles de requêtes/réponses
├── services/              # Logique métier
│   ├── category_service.py # Service des catégories
│   ├── poi_service.py     # Service des POI
│   ├── itinerary_service.py # Orchestrateur principal
│   └── osrm_service.py    # Service OSRM
├── pipeline/              # Pipeline de calcul
│   ├── itinerary_pipeline.py  # Pipeline complet
│   └── features/          # Composants du pipeline
│       ├── osrm.py       # Client OSRM asynchrone
│       ├── spatial_clustering.py  # Clustering géographique
│       ├── post_clustering.py     # Post-traitement
│       └── optimizer_*.py        # Algorithmes d'optimisation
├── repositories/          # Accès aux données
├── dependencies.py        # Injection de dépendances FastAPI
└── main.py               # Point d'entrée FastAPI
```

## Fonctionnalités Principales

### Gestion des Catégories
- **GET /main_categories** : Récupère toutes les catégories principales
- **POST /sub_categories** : Récupère les sous-catégories par catégories sélectionnées

### Points d'Intérêt (POI)
- **POST /poi/query** : Recherche avancée avec filtres multiples
- Filtrage par catégories, localisation, rayon
- Support des coordonnées géographiques et index H3

### Calcul d'Itinéraires Optimisés
- **POST /itinerary/compute** : Calcul d'itinéraires multi-jours
- **Clustering spatial** automatique par jour
- **Modes de transport** : walk, bike, car
- **Algorithmes** : NN2O, Génétique, Auto-sélection
- **Intégration OSRM** pour distances/temps réels

## Technologies Utilisées

### Backend
- **FastAPI** : Framework API moderne avec documentation auto-générée
- **Pydantic** : Validation et sérialisation des données
- **AsyncIO** : Programmation asynchrone

### Géospatial
- **OSRM** : Calcul d'itinéraires et distances
- **H3** : Indexation géospatiale hexagonale
- **PostGIS** : Extensions géographiques PostgreSQL

### Algorithmes
- **NN2O** : Nearest Neighbor 2-Opt (rapide)
- **Génétique** : Algorithme génétique (qualité)
- **Auto-sélection** : Choix automatique selon taille problème

## Pipeline de Calcul

Le pipeline suit 5 étapes principales :

1. **Clustering Spatial** : Regroupement géographique des POI par jour
2. **Préparation OSRM** : Formatage des coordonnées pour l'API OSRM
3. **Matrices OSRM** : Calcul asynchrone des distances/temps
4. **Optimisation** : Application du solveur sélectionné
5. **Enrichissement** : Ajout des métriques détaillées

## Modes de Transport

| Mode | Description | Cas d'usage |
|------|-------------|-------------|
| **walk** | Marche à pied | Centres villes, tourisme pédestre |
| **bike** | Vélo | Zones urbaines, pistes cyclables |
| **car** | Véhicule | Longues distances, zones rurales |

## Solveurs d'Optimisation

| Solveur | Performance | Qualité | Recommandation |
|---------|-------------|---------|----------------|
| **nn2o** | Très rapide | 🟊🟊🟊 | < 200 POI |
| **ga** | Lent | 🟊🟊🟊🟊🟊 | > 200 POI, qualité maximale |
| **auto** | Adaptatif | 🟊🟊🟊🟊 | Sélection automatique |

## Démarrage Rapide

### Installation
```bash
# Installation des dépendances
pip install fastapi uvicorn polars asyncpg psycopg2-binary

# Variables d'environnement
export OSRM_URL=http://localhost:5000
export POSTGRES_HOST=localhost
export POSTGRES_DB=vacances
```

### Lancement
```bash

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

### Documentation
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## 📡 Endpoints API

### Categories
```http
GET /main_categories
POST /sub_categories
```

### POI
```http
POST /poi/query
```

### Itinerary
```http
POST /itinerary/compute
```

## Exemple d'Utilisation

### Requête d'itinéraire
```json
{
  "pois": [
    {
      "poi_id": 123,
      "nom_du_poi": "Tour Eiffel",
      "latitude": 48.8584,
      "longitude": 2.2945,
      "main_category": "Monuments"
    }
  ],
  "days": 3,
  "transport_mode": "walk",
  "solver": "auto",
  "latitude": 48.85,
  "longitude": 2.35
}
```

### Réponse
```json
{
  "itinerary": [
    {
      "day": 1,
      "pois": [...],
      "total_distance_km": 2.3,
      "total_duration_min": 28
    }
  ],
  "trip_total_distance_km": 15.2,
  "trip_total_duration_min": 185,
  "optimizer": "nn2o"
}
```

## Configuration

### Variables d'environnement
```bash
# Base de données
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vacances
POSTGRES_USER=vacances_user
POSTGRES_PASSWORD=vacances_password

# OSRM
OSRM_URL=http://localhost:5000

# Application
LOG_LEVEL=INFO
DEBUG=false
```

## Développement

### Structure des services
- **Services** : Logique métier pure
- **Repositories** : Accès aux données
- **Pipeline** : Traitement complexe
- **API** : Routes FastAPI légères

### Tests
```bash
# Tests unitaires
pytest tests/unit/ # TODO

# Tests d'intégration
pytest tests/integration/ # TODO

# Tests API
pytest tests/api/ # TODO
```

### Debug
```bash
# Mode debug
uvicorn app.main:app --reload --log-level debug

# Logs du pipeline
export LOG_LEVEL=DEBUG
```

## Performance

### Optimisations
- **Polars** : DataFrames optimisées vs Pandas
- **AsyncIO** : Requêtes OSRM parallèles
- **Clustering** : Réduction complexité algorithmique
- **Cache** : Matrices de distances réutilisées

### Recommandations
- **< 50 POI** : Mode auto, traitement instantané
- **50-200 POI** : NN2O recommandé
- **> 200 POI** : GA pour meilleure qualité

## Docker

### Déploiement avec Docker Compose
```bash
# Depuis le répertoire docker/api/
cd docker/api/
./deploy.sh
```

Ou manuellement :
```bash
cd docker/api/
docker-compose up --build -d
```

### Services inclus
- **API FastAPI** : http://localhost:8000
- **OSRM Backend** : http://localhost:5000  
- **PostgreSQL** : localhost:5433
- **Documentation** : http://localhost:8000/docs

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
services:
  api:
    build:
      context: ..
      dockerfile: docker/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OSRM_URL=http://osrm-backend:5000
      - POSTGRES_HOST=postgres-vacances
```

##  Dépannage

### Problèmes courants
- **OSRM timeout** : Vérifier `OSRM_URL` et connectivité
- **Memory error** : Réduire nombre POI ou utiliser NN2O
- **Database connection** : Vérifier variables POSTGRES_*

### Logs utiles
```bash
# Logs FastAPI
tail -f logs/app.log

# Logs pipeline
grep "Pipeline" logs/app.log

# Logs OSRM
docker logs osrm-container
```


## Notes Techniques

### Pipeline service
- **Dependency Injection** : FastAPI `Depends()`
- **Service Layer** : Séparation logique métier
- **Repository Pattern** : Abstraction données
- **Pipeline Pattern** : Traitement par étapes

### Async/Await
- OSRM client asynchrone pour performances
- Support des requêtes concurrentes

---

