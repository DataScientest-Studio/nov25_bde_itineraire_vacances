# Infrastructure Docker TripMaNGo - Déploiement et monitoring simplifiés

Infrastructure complète pour le déploiement de l'API TripMaNGo avec monitoring intégré.

---

## Architecture

### Services Principaux
- **API FastAPI** : Service principal de calcul d'itinéraires
- **PostgreSQL Vacances** : Base de données métier avec PostGIS
- **PostgreSQL Airflow** : Base de données pour l'orchestration
- **OSRM** : Trois instances (car, bike, foot) pour le routing
- **Airflow** : Orchestration des ETL et pipelines
- **pgAdmin** : Interface d'administration des BDD

### Services de Monitoring
- **Prometheus** : Collecteur de métriques
- **Grafana** : Visualisation et dashboards
- **AlertManager** : Gestion des alertes
- **Node Exporter** : Métriques système
- **cAdvisor** : Métriques des conteneurs Docker

---

## Démarrage Rapide

### 1. Infrastructure sans monitoring

1. Créer le fichier de config. avec les variables d'environnement :
   ```bash
   cp .env.example .env
   ````
2. Lancer la compilation des fichiers OSRM :
   ```bash
   COMPOSE_PROFILES=osrm_compilation docker compose up --build
   ````
   Cette étape dure 60 minutes à peu prés et permet de préparer les fichiers nécessaires aux 3 serveurs OSRM(car, foot, bike).
   
3. Lancer les services "infrastructure" : airflow, base de données postgresql, 3 serveurs OSRM et le reverse proxy nginx  :
   ```bash
   COMPOSE_PROFILES=infrastructure docker compose up --build -d
   ````
   Suivre les étapes d'itinialisation de la base de données décrites [ici](../dags/README.md) à partir de l'étape : "Accès à l'Interface Airflow".
   
5. Lancer les services "app" : api et streamlit 
   ```bash
   COMPOSE_PROFILES=app docker compose up --build -d
   ````

#### Vérification des services
```bash
docker-compose ps
```

### 2. Infrastructure avec Monitoring
- Déployer l'infrastructure (1).
- Ajouter le  service de monitoring :
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. API Seulement (Développement)
```bash
cd docker/api/
docker-compose up -d
```

---

## Accès aux Services

### Services Principaux
| Service | URL | Identifiants |
|---------|-----|-------------|
| API TripMaNGo | http://localhost:8000 | - |
| Documentation API | http://localhost:8000/docs | - |
| PostgreSQL Vacances | localhost:5433 | vacances_user/vacances_password |
| PostgreSQL Airflow | localhost:5434 | airflow/airflow |
| pgAdmin | http://localhost:5050 | admin@admin.com/admin123 |
| Airflow WebUI | http://localhost:8080 | airflow/airflow |
| OSRM Car | http://localhost:5000 | - |
| OSRM Bike | http://localhost:5002 | - |
| OSRM Foot | http://localhost:5001 | - |

### Services de Monitoring [Lire la Doc Monitoring](./monitoring/README.md)
| Service | URL | Identifiants |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin123 |
| AlertManager | http://localhost:9093 | - |
| Node Exporter | http://localhost:9100/metrics | - |
| cAdvisor | http://localhost:8080 | - |

---

## Configuration

### Variables d'Environnement
Créer un fichier `.env` à la racine du projet :

```bash
# Base de données Vacances
POSTGRES_VACANCES_HOST=postgres-vacances
POSTGRES_VACANCES_PORT=5432
POSTGRES_VACANCES_DB=vacances
POSTGRES_VACANCES_USER=vacances_user
POSTGRES_VACANCES_PASSWORD=vacances_password
POSTGRES_VACANCES_EXTERNAL_PORT=5433

# Base de données Airflow
POSTGRES_AIRFLOW_USER=airflow
POSTGRES_AIRFLOW_PASSWORD=airflow
POSTGRES_AIRFLOW_DB=airflow

# Airflow
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.

# Notifications (optionnel)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

---

## Monitoring

### Prometheus
- **Configuration** : `monitoring/prometheus/prometheus.yml`
- **Règles d'alertes** : `monitoring/prometheus/alert_rules.yml`
- **Rétention** : 30 jours
- **Scraping** : 15s (API), 30s (BDD), 30s (système)

### Grafana
- **Provisioning automatique** : Datasources et dashboards
- **Plugins préinstallés** : Clock panel, Simple JSON
- **Dashboards disponibles** :
  - Overview TripMaNGo
  - API Performance
  - Infrastructure
  - Base de données

### Alertes Configurées
- **API** : Latence > 2s, taux d'erreur > 10%
- **Base de données** : Service down, connexions > 80
- **OSRM** : Service down, latence > 5s
- **Système** : CPU > 80%, RAM > 85%, disque > 90%
- **Airflow** : Scheduler lag, échecs de tâches

---

## Gestion des Services

### Commandes Utiles
```bash
# Démarrer tous les services
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Arrêter tous les services
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down

# Voir les logs
docker-compose logs -f api
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Reconstruire une image
docker-compose build api
docker-compose build --no-cache prometheus

# Redémarrer un service
docker-compose restart api
docker-compose restart postgres-vacances

# Nettoyer les volumes (attention)
docker-compose down -v
```

### Maintenance
```bash
# Sauvegarder les données PostgreSQL
docker exec postgres-vacances pg_dump -U vacances_user vacances > backup.sql

# Restaurer les données
docker exec -i postgres-vacances psql -U vacances_user vacances < backup.sql

# Nettoyer les images Docker non utilisées
docker image prune -f

# Vider les logs
docker system prune -f
```

---

## Dépannage

### Problèmes Courants

**Services ne démarrent pas**
```bash
# Vérifier les ports utilisés
netstat -tulpn | grep :8000

# Vérifier les conflits de réseaux
docker network ls
docker network prune
```

**Base de données inaccessible**
```bash
# Vérifier le statut des conteneurs
docker-compose ps

# Vérifier les logs PostgreSQL
docker-compose logs postgres-vacances

# Se connecter manuellement
docker exec -it postgres-vacances psql -U vacances_user -d vacances
```

**Monitoring ne collecte pas les métriques**
```bash
# Vérifier la configuration Prometheus
docker exec prometheus cat /etc/prometheus/prometheus.yml

# Vérifier les targets
curl http://localhost:9090/api/v1/targets

# Tester les endpoints de métriques
curl http://localhost:8000/metrics
curl http://localhost:9187/metrics
```

### Performance

**Optimisation Docker**
```bash
# Optimiser les volumes
docker-compose down
docker volume prune
```

**Monitoring Performance**
```bash
# Vérifier l'utilisation des ressources
docker stats

# Optimiser Prometheus
# Réduire la rétention dans prometheus.yml
# Ajuster les intervalles de scraping
```

---

## Structure des Fichiers

```
docker/
├── docker-compose.yml              # Infrastructure principale
├── docker-compose.monitoring.yml   # Services de monitoring
├── api/                            # Service API
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── deploy.sh
├── airflow/                        # Service Airflow
│   ├── Dockerfile
│   └── docker-compose.yml
├── osrm/                           # Configuration OSRM
│   ├── nginx.conf
│   └── scripts/
├── streamlit/                      # Dashboard Streamlit (optionnel)
└── monitoring/                     # Configuration monitoring
    ├── prometheus/
    │   ├── Dockerfile
    │   ├── prometheus.yml
    │   └── alert_rules.yml
    ├── grafana/
    │   └── provisioning/
    │       ├── datasources/
    │       └── dashboards/
    └── alertmanager.yml
```

---

## Sécurité

### Bonnes Pratiques
- **Changer les mots de passe par défaut**
- **Utiliser des secrets Docker**
- **Limiter l'exposition des ports**
- **Activer HTTPS en production**
- **Configurer les firewalls**

### Variables Sensibles
```bash
# Utiliser Docker secrets
echo "your_password" | docker secret create postgres_password -

# Ou utiliser un fichier .env non versionné
.env
.env.local
```

---

## Documentation Complémentaire

- **API** : [../app/README.md](../app/README.md)
- **Services** : [../app/services/README.md](../app/services/README.md)
- **Pipeline** : [../app/pipeline/README.md](../app/pipeline/README.md)
- **Benchmark** : [../src/benchmark/README.md](../src/benchmark/README.md)

---


