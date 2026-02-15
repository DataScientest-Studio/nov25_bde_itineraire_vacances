# API TripMaNGo - Itinéraires de Vacances Intelligents

API FastAPI pour le calcul d'itinéraires de vacances optimisés avec clustering spatial et algorithmes de résolution.

---

## Architecture

L'API est organisée en couches claires avec séparation des responsabilités :

```
app/
├── api/                    # Routes HTTP (Couche Présentation)
│   ├── categories.py       # Gestion des catégories POI
│   ├── poi.py             # Points d'intérêt
│   └── itinerary.py        # Calcul d'itinéraires
├── services/              # Logique Métier (Couche Service)
│   ├── category_service.py # Service des catégories
│   ├── poi_service.py     # Service des POI
│   ├── itinerary_service.py # Orchestrateur principal
│   └── osrm_service.py    # Service OSRM
├── pipeline/              # Pipeline Algorithmique (Couche Calcul)
│   ├── itinerary_pipeline.py  # Pipeline complet
│   └── features/          # Composants du pipeline
│       ├── osrm.py       # Client OSRM asynchrone
│       ├── spatial_clustering.py  # Clustering géographique
│       ├── post_clustering.py     # Post-traitement
│       └── optimizer_*.py        # Algorithmes d'optimisation
├── models/                # Modèles de Données (Couche Contrat)
│   ├── categories.py       # Catégories et sous-catégories
│   ├── poi.py             # Structure des POI
│   └── itinerary.py        # Modèles de requêtes/réponses
├── core/                  # Configuration Infrastructure
│   └── database.py        # Gestionnaire de base de données
├── dependencies.py        # Injection de Dépendances
└── main.py               # Point d'entrée FastAPI
```

---

## Fonctionnalités Principales

### Points d'Intérêt (POI)
- **Recherche avancée** : Filtrage par catégories, localisation, score
- **Indexation géospatiale** : Support des coordonnées GPS et index H3
- **Métadonnées riches** : Descriptions, contacts, informations pratiques

### Gestion des Catégories
- **Catégories principales** : Hiérarchie complète des thématiques
- **Sous-catégories** : Classification fine des points d'intérêt
- **Filtrage multi-niveaux** : Sélection flexible par besoins

### alcul d'Itinéraires Optimisés
- **Clustering spatial automatique** : Répartition intelligente par jour
- **Modes de transport** : Marche, vélo, voiture avec profils OSRM
- **Algorithmes d'optimisation** : NN2O (rapide), Génétique (qualité), Auto (adaptatif)
- **Enrichissement complet** : Distances, durées, métriques détaillées

---

## Services - Couche Métier

### ItineraryService - Orchestrateur Principal

**Rôle** : Transformation des requêtes API en itinéraires optimisés

**Flux d'exécution** :
```
POIs API → DataFrame Polars → Pipeline Algorithmique → Enrichissement → Réponse JSON
```

**Méthode clé** :
```python
async def compute_itinerary(
    pois: List[POI],
    days: int,
    transport_mode: str,
    solver: str,
    start_lat: float,
    start_lon: float
) -> Dict[str, Any]
```

**Retour** : Itinéraire structuré par jour avec POIs enrichis, métriques et géométries OSRM

*Documentation complète : [services/README.md](./services/README.md)*

---

## Pipeline Algorithmique

### ItineraryPipeline - Cœur de Calcul

**Architecture en 6 étapes** :

1. **Clustering Spatial** : Regroupement géographique des POIs par jour avec index H3
2. **Post-Clustering** : Rééquilibrage intelligent (restaurants, diversité, densité)
3. **Préparation OSRM** : Formatage des coordonnées et filtrage par mode de transport
4. **Matrices OSRM** : Calcul asynchrone des distances et durées
5. **Solveurs d'Optimisation** : NN2O, Génétique ou sélection automatique
6. **Enrichissement** : Ajout des métriques détaillées (cumuls, totaux, etc.)

*Documentation complète : [pipeline/README.md](./pipeline/README.md)*

---

## Algorithmes d'Optimisation

| Solveur | Performance | Qualité | Recommandation | Seuil AUTO |
|---------|-------------|---------|----------------|------------|
| **NN2O** | Très rapide (< 100ms) | 🟊🟊🟊 | Petits clusters | ≤ 6 POIs |
| **Génétique** | Lent (1-3s) | 🟊🟊🟊🟊🟊 | Grands clusters | > 6 POIs |
| **Auto** | Adaptatif | 🟊🟊🟊🟊 | Sélection automatique | Intelligent |

*Analyse complète : [benchmark/README.md](../src/benchmark/README.md)*

---

## Modes de Transport

| Mode | Profil OSRM | Rayon Max | Cas d'usage |
|------|-------------|-----------|-------------|
| **walk** | foot | 14 km | Centres villes, tourisme pédestre |
| **bike** | bike | 27 km | Zones urbaines, pistes cyclables |
| **car** | driving | 40 km | Longues distances, zones rurales |

---

## Endpoints API

### Catégories
```http
GET  /main_categories          # Toutes les catégories principales
POST /sub_categories          # Sous-catégories par sélection
```

### Points d'Intérêt
```http
POST /poi/query               # Recherche avancée avec filtres
```

### Itinéraires
```http
POST /itinerary/compute       # Calcul d'itinéraire optimisé
```

### Documentation Interactive
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

---

## Technologies

### Backend
- **FastAPI** : Framework API moderne avec validation automatique
- **Pydantic** : Contrats de données et validation
- **AsyncIO** : Programmation asynchrone haute performance

### Géospatial
- **OSRM** : Calcul d'itinéraires et distances en temps réel
- **H3** : Indexation géospatiale hexagonale (Uber)
- **PostGIS** : Extensions géographiques PostgreSQL

### Algorithmique
- **Polars** : DataFrames optimisés vs Pandas
- **NumPy** : Calculs matriciels performants
- **DEAP** : Algorithmes génétiques

---

## Démarrage Rapide

### Installation
```bash
# Installation des dépendances
pip install fastapi uvicorn polars psycopg2-binary

# Variables d'environnement
export OSRM_URL=http://localhost:5000
export POSTGRES_HOST=localhost
export POSTGRES_DB=vacances
```

### Lancement
```bash
# Développement avec rechargement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

# Construction et lancement à partir du script
```bash
./deploy.sh
```

```bash
# Construction et lancement à aprtir de docker-compose
docker-compose up -d
```

---

## Performance

### Métriques
- **< 50 POIs** : Traitement instantané (< 1s)
- **50-200 POIs** : Optimisation rapide (1-5s)
- **> 200 POIs** : Calcul intensif (5-30s)

### Optimisations
- **Async OSRM** : Requêtes parallèles
- **Clustering** : Réduction complexité algorithmique
- **Cache H3** : Indexation géospatiale
- **Polars** : DataFrames haute performance

---

## Configuration

### Variables d'Environnement
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

### Dépendances
Voir [requirements.txt](./requirements.txt) pour la liste complète des packages.

---

## Dépannage

### Problèmes Courants
- **OSRM timeout** : Vérifier `OSRM_URL` et connectivité
- **Memory error** : Réduire nombre POIs ou utiliser NN2O
- **Database connection** : Vérifier variables `POSTGRES_*`

### Logs Utiles
```bash
# Logs application
tail -f logs/app.log

# Logs pipeline
grep "Pipeline" logs/app.log

# Logs OSRM
docker logs osrm-container
```

---

## Documentation Complète

- **Services** : [services/README.md](./services/README.md) - Couche métier
- **Pipeline** : [pipeline/README.md](./pipeline/README.md) - Algorithmique
- **Benchmark** : [../src/benchmark/README.md](../src/benchmark/README.md) - Performance

---
[Retour sur la documentation principale](../README.md)