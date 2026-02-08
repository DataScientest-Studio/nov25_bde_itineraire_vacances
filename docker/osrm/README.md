# 🗺️ OSRM Multi-Régions & Multi-Profils - Service de Routage

## Configuration

Ce service OSRM couvre 3 régions françaises avec **3 profils de transport** :
- **Régions** : Auvergne Rhône-Alpes, Île-de-France, Bretagne
- **Profils** : car (véhicule), bike (vélo), walk (piéton)

## Architecture Dockerisée

```
┌─────────────────┐
│   OSRM          │
│   Multi-Profils │
│   Port 5000     │
└─────────────────┘
         │
    Données combinées
    (3 régions × 3 profils)
```

## Profils Disponibles

### 🚗 Car (Véhicule)
- **Utilisation** : Routes pour voitures, motos
- **Type de routes** : Autoroutes, routes nationales, voies rapides
- **Optimisation** : Temps de trajet minimum
- **Poids** : Priorité aux routes rapides

### 🚴 Bike (Vélo)  
- **Utilisation** : Cyclistes, vélos électriques
- **Type de routes** : Pistes cyclables, voies dédiées, rues calmes
- **Optimisation** : Sécurité et confort
- **Poids** : Évitement des routes à grande circulation

### 🚶 Walk (Piéton)
- **Utilisation** : Piétons, marcheurs, trottinettes
- **Type de routes** : Trottoirs, passages piétons, zones piétonnes
- **Optimisation** : Distance et sécurité
- **Poids** : Chemins les plus directs et sécurisés

## Déploiement

### 1. Prérequis
```bash
# Docker et Docker Compose installés
docker --version
docker-compose --version
```

### 2. Lancement
```bash
# Rendre le script exécutable
chmod +x deploy.sh

# Lancer le déploiement
./deploy.sh
```

### 3. Accès au service
- **OSRM Backend** : http://localhost:5000
- **Health Check** : http://localhost:5000/health

### 4. Changer de profil
Par défaut, le service utilise le profil `car`. Pour changer :

```bash
# Arrêter le service
docker-compose down

# Démarrer avec un autre profil
OSRM_PROFILE=bike docker-compose up -d    # Pour le vélo
OSRM_PROFILE=walk docker-compose up -d    # Pour les piétons
OSRM_PROFILE=car docker-compose up -d     # Pour les voitures (défaut)
```

### 5. Exemples d'utilisation
```bash
# Voir tous les exemples
chmod +x examples_profils.sh
./examples_profils.sh
```

## API Endpoints

### Calcul d'itinéraire par profil

#### 🚗 Profil Car (Véhicule)
```bash
# Route voiture entre deux points
curl "http://localhost:5000/route/v1/car/2.35,48.85;2.36,48.86?overview=false"

# Route voiture avec alternatives
curl "http://localhost:5000/route/v1/car/2.35,48.85;2.36,48.86?alternatives=true"
```

#### 🚴 Profil Bike (Vélo)
```bash
# Route vélo entre deux points
curl "http://localhost:5000/route/v1/bike/2.35,48.85;2.36,48.86?overview=false"

# Route vélo avec détails des étapes
curl "http://localhost:5000/route/v1/bike/2.35,48.85;2.36,48.86?steps=true"
```

#### 🚶 Profil Walk (Piéton)
```bash
# Route piétonne entre deux points
curl "http://localhost:5000/route/v1/walk/2.35,48.85;2.36,48.86?overview=false"

# Route piétonne avec détails
curl "http://localhost:5000/route/v1/walk/2.35,48.85;2.36,48.86?steps=true"
```

### Matrice de distances
```bash
# Matrice de distances
curl "http://localhost:5000/table/v1/driving/2.35,48.85;2.36,48.86;2.37,48.87"

# Matrice avec annotations
curl "http://localhost:5000/table/v1/driving/2.35,48.85;2.36,48.86?annotations=distance,duration"
```

### Recherche d'itinéraire
```bash
# Matching de trace GPS
curl "http://localhost:5000/matching/v1/driving/2.35,48.85;2.36,48.86;2.37,48.87"
```

### Recherche de proximité
```bash
# Points à proximité
curl "http://localhost:5000/nearest/v1/driving/2.35,48.85"
```

## Configuration des Régions

### Données utilisées
- **Source** : Geofabrik (OpenStreetMap)
- **Format** : PBF (Protocolbuffer Binary Format)
- **Mise à jour** : Quotidienne

### Taille des données
- **Auvergne Rhône-Alpes** : ~800 MB
- **Île-de-France** : ~200 MB
- **Bretagne** : ~300 MB
- **Total combiné** : ~1.3 GB (compressé)

### Ajouter une région
1. Modifier `prepare_osrm_data.sh`
2. Ajouter la région dans le tableau `REGIONS`
3. Reconstruire l'image

```bash
# Exemple pour ajouter PACA
["provence-alpes-cote-dazur"]="https://download.geofabrik.de/europe/france/provence-alpes-cote-dazur-latest.osm.pbf"
```

## Gestion du Service

### Voir les logs
```bash
# Logs en temps réel
docker-compose logs -f

# Logs du service OSRM
docker-compose logs -f osrm-multi-regions
```

### Redémarrer le service
```bash
docker-compose restart osrm-multi-regions
```

### Mettre à jour les données
```bash
# Supprimer les anciennes données
docker volume rm osrm_osrm_data

# Relancer avec nouvelles données
./deploy.sh
```

### Arrêter le service
```bash
docker-compose down
```

## Intégration avec votre App

### Configuration FastAPI
Dans votre `app/dependencies.py` :

```python
def get_itinerary_service() -> ItineraryService:
    return ItineraryService(
        osrm_url=os.getenv("OSRM_URL", "http://osrm-multi-regions:5000"),
    )
```

### Variables d'environnement
```yaml
# Dans docker-compose.yml de l'API
environment:
  - OSRM_URL=http://osrm-multi-regions:5000
```

## Performance

### Spécifications recommandées
- **CPU** : 4+ cœurs
- **RAM** : 8+ GB
- **Stockage** : 10+ GB SSD

### Optimisation
- Utilisez l'algorithme MLD (Multi-Level Dijkstra)
- Cache activé par défaut
- Support des requêtes concurrentes

## Dépannage

### Le service ne démarre pas
```bash
# Vérifier les logs
docker-compose logs osrm-multi-regions

# Vérifier l'espace disque
df -h

# Recréer depuis zéro
docker-compose down -v
./deploy.sh
```

### Erreur de mémoire
```bash
# Augmenter la mémoire Docker
# Dans Docker Desktop : Settings > Resources > Memory

# Ou limiter les processus concurrents
# Modifier le CMD dans le Dockerfile
```

### Test de connexion
```bash
# Test simple
curl -f http://localhost:5000/route/v1/driving/2.35,48.85;2.36,48.86

# Test avec une ville de chaque région
# Lyon (Auvergne Rhône-Alpes)
curl "http://localhost:5000/route/v1/driving/4.85,45.75;4.86,45.76"

# Paris (Île-de-France)
curl "http://localhost:5000/route/v1/driving/2.35,48.85;2.36,48.86"

# Rennes (Bretagne)
curl "http://localhost:5000/route/v1/driving/-1.68,48.11;-1.67,48.12"
```

## Surveillance

### Métriques disponibles
- Nombre de requêtes/secondes
- Temps de réponse moyen
- Taux d'erreur
- Utilisation mémoire/CPU

### Health Check
```bash
# Vérifier le statut
curl http://localhost:5000/health

# Vérifier le statut du container
docker-compose ps
```
