# 🚀 API FastAPI - Itinéraires de Vacances

## Architecture Dockerisée

L'API FastAPI est composée de :
- **API FastAPI** : Service REST sur le port 8000
- **OSRM Backend** : Service de calcul d'itinéraires sur le port 5000
- **PostgreSQL + PostGIS** : Base de données géospatiale sur le port 5433

## Déploiement Rapide

### 1. Prérequis
```bash
# Docker et Docker Compose installés
docker --version
docker-compose --version
```

### 2. Lancement depuis le dossier docker/api/
```bash
# Rendre le script exécutable
chmod +x deploy.sh

# Lancer le déploiement
./deploy.sh
```

### 3. Accès aux services
- **API FastAPI** : http://localhost:8000
- **Documentation Swagger** : http://localhost:8000/docs
- **OSRM Backend** : http://localhost:5000
- **PostgreSQL** : localhost:5433

## Endpoints Principaux

### Itinéraires
- `POST /itinerary/compute` : Calculer un itinéraire optimisé
- `GET /itinerary/status/{id}` : Statut d'un calcul

### Points d'Intérêt
- `GET /poi/` : Lister les POIs
- `GET /poi/{id}` : Détails d'un POI
- `GET /poi/categories` : Catégories disponibles

## Configuration

### Variables d'environnement
- `OSRM_URL` : URL du service OSRM (défaut: http://osrm-backend:5000)
- `POSTGRES_HOST` : Hôte PostgreSQL (défaut: postgres-vacances)
- `POSTGRES_DB` : Nom de la base (défaut: vacances)
- `POSTGRES_USER` : Utilisateur (défaut: vacances_user)
- `POSTGRES_PASSWORD` : Mot de passe (défaut: vacances_password)

## Développement

### Mode développement
```bash
# Lancement avec rechargement automatique
docker-compose up api

# Les modifications dans src/ sont automatiquement reflétées
```

### Tests
```bash
# Accéder au container
docker exec -it itinerary-api bash

# Lancer les tests
python -m pytest tests/
```

## Gestion des Services

### Voir les logs
```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f api
docker-compose logs -f osrm-backend
```

### Redémarrer un service
```bash
docker-compose restart api
```

### Arrêter les services
```bash
docker-compose down
```

## Base de Données

### Connexion
```bash
# Avec psql
psql -h localhost -p 5433 -U vacances_user -d vacances

# Avec pgAdmin (si configuré)
# http://localhost:5050
```

### Sauvegarde
```bash
# Exporter la base
docker exec postgres-vacances pg_dump -U vacances_user vacances > backup.sql

# Importer la base
docker exec -i postgres-vacances psql -U vacances_user vacances < backup.sql
```

## Dépannage

### L'API ne se connecte pas à la base
```bash
# Vérifier la connexion
docker exec -it itinerary-api python -c "
from app.core.database import db_manager
conn = db_manager.get_conn()
print('Connexion OK')
db_manager.return_conn(conn)
"
```

### OSRM ne répond pas
```bash
# Tester le service
curl http://localhost:5000/route/v1/driving/2.35,48.85;2.36,48.86

# Recréer les données OSRM
docker-compose run --rm osrm-data
```

### L'API ne démarre pas
```bash
# Vérifier les logs détaillés
docker-compose logs api

# Vérifier la configuration
docker exec -it itinerary-api python -c "
import os
print('OSRM_URL:', os.getenv('OSRM_URL'))
print('POSTGRES_HOST:', os.getenv('POSTGRES_HOST'))
"
```
