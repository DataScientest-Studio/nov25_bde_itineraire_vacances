# Monitoring TripMaNGo - Guide de Déploiement

Infrastructure de monitoring complète pour l'application TripMaNGo avec Prometheus, Grafana et exporters.

---

## Architecture du Monitoring

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   API FastAPI   │ ──▶ │   Prometheus     │ ──▶│    Grafana      │
│   :8000/metrics │     │   :9090          │     │   :3000         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   OSRM Exporter │     │ PostgreSQL Exporter │     │  Visualisation  │
│   :8001/metrics │     │   :9187/metrics     │     │   Dashboards    │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────────┐
│   Streamlit     │     │   Airflow          │
│   :8501         │     │   :8080            │
└─────────────────┘     └─────────────────────┘
```

---

## Déploiement Rapide

### 1. Démarrer l'Infrastructure Complète

```bash
cd docker/monitoring
docker-compose up -d
```

### 2. Vérifier les Services
```bash
# Vérifier tous les conteneurs
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Vérifier les métriques API
curl http://localhost:8000/metrics

# Vérifier Prometheus targets
curl http://localhost:9090/api/v1/targets
```

---

## Accès aux Services

| Service | URL | Login | Description |
|---------|------|--------|-------------|
| **Grafana** | http://localhost:3000 | admin/admin123 | Dashboards et visualisation |
| **Prometheus** | http://localhost:9090 | - | Collecteur de métriques |
| **API Metrics** | http://localhost:8000/metrics | - | Métriques FastAPI |
| **OSRM Exporter** | http://localhost:8001/metrics | - | Métriques OSRM |
| **Postgres Exporter** | http://localhost:5050 | admin@admin.com/admin | Administration BDD |

---

## Métriques Disponibles

### API FastAPI
- `http_requests_total` : Nombre total de requêtes HTTP
- `http_request_duration_seconds` : Temps de réponse des requêtes
- `active_connections` : Connexions actives
- `itinerary_requests_total` : Requêtes de calcul d'itinéraire
- `itinerary_computation_duration_seconds` : Temps de calcul
- `itinerary_pois_processed` : Nombre de POIs traités

### OSRM Exporter
- `osrm_requests_total` : Requêtes OSRM par profil (car/bike/foot)
- `osrm_request_duration_seconds` : Temps de réponse OSRM
- `osrm_active_profiles` : Profils OSRM actifs
- `osrm_health_status` : État de santé des services OSRM

### PostgreSQL
- `pg_stat_activity_count` : Connexions actives
- `pg_database_size_bytes` : Taille des bases de données
- `pg_stat_database_tup_returned` : Tuples retournées
- `pg_locks_count` : Verrous en cours

---

## Dashboards Grafana

### Dashboard Principal : TripMaNGO Overview
- **Panneaux** : 8 graphiques temps réel
- **Rafraîchissement** : 5 secondes
- **Période par défaut** : Dernière heure

**Graphiques disponibles** :
1. Taux de requêtes API (par endpoint)
2. Temps de réponse API (50th/95th percentile)
3. Connexions actives PostgreSQL
4. Requêtes d'itinéraire (par transport/solveur)
5. Temps de calcul d'itinéraire
6. Métriques OSRM par profil
7. Taille base de données
8. État des services Airflow

---

## Configuration

### Variables d'Environnement
```bash
# Configuration Prometheus
PROMETHEUS_RETENTION=30d
PROMETHEUS_SCRAPE_INTERVAL=15s

# Configuration Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin123
GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource

# Configuration Base de données
POSTGRES_VACANCES_HOST=postgres-vacances
POSTGRES_VACANCES_PORT=5432
```

### Fichiers de Configuration
- `prometheus/prometheus.yml` : Configuration scrape Prometheus
- `grafana/provisioning/datasources/` : Sources de données auto-configurées
- `grafana/dashboards/` : Dashboards auto-provisionnés
- `docker-compose.yml` : Services monitoring

---

## Maintenance

### Logs
```bash
# Logs Prometheus
docker logs prometheus -f

# Logs Grafana
docker logs grafana -f

# Logs OSRM Exporter
docker logs osrm-exporter -f

# Logs API
docker logs itinerary-api -f
```

### Nettoyage
```bash
# Arrêter le monitoring
docker-compose -f docker-compose.yml down

# Nettoyer les volumes (attention!)
docker-compose -f docker-compose.yml down -v

# Nettoyage complet Docker
docker system prune -f
```

---

## Personnalisation

### Ajouter un Dashboard Grafana
1. Se connecter à Grafana (admin/admin123)
2. Importer un dashboard JSON
3. Personnaliser les requêtes PromQL
4. Configurer les visualisations

---

## Dépannage

# Vérifier les logs
```bash
docker logs prometheus
docker logs grafana
```

**Métriques non collectées**
```bash
# Vérifier les targets Prometheus
curl http://localhost:9090/api/v1/targets

# Tester les endpoints de métriques
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
```

**Problèmes réseau**
```bash
# Vérifier le réseau Docker
docker network ls
docker network inspect docker_vacances-network

# Recréer le réseau si nécessaire
docker network create docker_vacances-network
```

---

[Retour sur la documentation Docker](../README.md)

