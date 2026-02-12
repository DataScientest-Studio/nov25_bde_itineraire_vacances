# Monitoring TripMaNGo - Guide de Déploiement

Infrastructure complète de monitoring pour l'API TripMaNGo avec Prometheus et Grafana.

---

## 🎯 Architecture du Monitoring

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API FastAPI  │───▶│   Prometheus    │───▶│    Grafana     │
│   :8000/metrics│    │   :9090        │    │   :3000        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OSRM Exporter│    │ PostgreSQL Exporter│    │  AlertManager  │
│   :8001/metrics│    │   :9187/metrics │    │   :9093        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🚀 Déploiement Rapide

### 1. Démarrer l'Infrastructure Principale
```bash
cd docker/
docker-compose up -d
```

### 2. Démarrer le Monitoring Complet
```bash
docker-compose -f docker-compose.monitoring-simple.yml up -d
```

### 3. Vérifier les Services
```bash
# Vérifier tous les conteneurs
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Vérifier les métriques API
curl http://localhost:8000/metrics

# Vérifier Prometheus targets
curl http://localhost:9090/api/v1/targets
```

---

## 📊 Accès aux Services

| Service | URL | Login | Description |
|---------|------|--------|-------------|
| **Grafana** | http://localhost:3000 | admin/admin123 |
| **Prometheus** | http://localhost:9090 | - |
| **API Metrics** | http://localhost:8000/metrics | - |
| **OSRM Exporter** | http://localhost:8001/metrics | - |
| **AlertManager** | http://localhost:9093 | - |

---

## 📈 Métriques Disponibles

### API FastAPI
- `http_requests_total` : Nombre total de requêtes HTTP
- `http_request_duration_seconds` : Temps de réponse des requêtes
- `active_connections` : Connexions actives
- `itinerary_requests_total` : Requêtes de calcul d'itinéraire
- `itinerary_computation_duration_seconds` : Temps de calcul
- `itinerary_pois_processed` : Nombre de POIs traités

### OSRM Exporter
- `osrm_requests_total` : Requêtes OSRM par profil
- `osrm_request_duration_seconds` : Temps de réponse OSRM
- `osrm_active_profiles` : Profils OSRM actifs

### PostgreSQL
- `pg_stat_activity_count` : Connexions actives
- `pg_database_size_bytes` : Taille des bases de données
- `pg_stat_database_tup_returned` : Tuples retournées

---

## 🎨 Dashboards Grafana

### Dashboard Principal : TripMaNGo Overview
- **Panneaux** : 8 graphiques temps réel
- **Rafraîchissement** : 5 secondes
- **Période par défaut** : Dernière heure

**Graphiques disponibles** :
1. Taux de requêtes API (par endpoint)
2. Temps de réponse API (50th/95th percentile)
3. Connexions actives
4. Taux d'erreur API
5. Requêtes d'itinéraire (par transport/solveur)
6. Temps de calcul d'itinéraire
7. Connexions PostgreSQL
8. Taille base de données

---

## 🚨 Alertes Configurées

### Alertes API
- **Latence élevée** : 95th percentile > 2s (warning)
- **Taux d'erreur** : Erreurs 5xx > 10% (critical)

### Alertes Infrastructure
- **Service down** : Any service unavailable (critical)
- **CPU élevé** : > 80% (warning)
- **Mémoire élevée** : > 85% (warning)
- **Disque plein** : > 90% (critical)

---

## 🔧 Configuration

### Variables d'Environnement
```bash
# Configuration Prometheus
PROMETHEUS_RETENTION=30d
PROMETHEUS_SCRAPE_INTERVAL=15s

# Configuration Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin123
GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
```

### Fichiers de Configuration
- `prometheus/prometheus.yml` : Configuration scrape Prometheus
- `prometheus/alert_rules.yml` : Règles d'alertes
- `grafana/provisioning/datasources/` : Sources de données auto-configurées
- `grafana/dashboards/` : Dashboards auto-provisionnés

---

## 🛠️ Maintenance

### Redémarrer les Services
```bash
# Redémarrer tout le monitoring
docker-compose -f docker-compose.monitoring-simple.yml restart

# Redémarrer un service spécifique
docker-compose -f docker-compose.monitoring-simple.yml restart prometheus
docker-compose -f docker-compose.monitoring-simple.yml restart grafana
```

### Logs
```bash
# Logs Prometheus
docker logs prometheus -f

# Logs Grafana
docker logs grafana -f

# Logs OSRM Exporter
docker logs osrm-exporter -f
```

### Nettoyage
```bash
# Arrêter le monitoring
docker-compose -f docker-compose.monitoring-simple.yml down

# Nettoyer les volumes (attention!)
docker-compose -f docker-compose.monitoring-simple.yml down -v
```

---

## 📊 Personnalisation

### Ajouter des Métriques API
1. Importer `prometheus_client` dans votre code
2. Définir des compteurs/histogrammes/gauges
3. Exposer via endpoint `/metrics`
4. Ajouter le scraping dans `prometheus.yml`

### Créer des Dashboards Grafana
1. Se connecter à Grafana (admin/admin123)
2. Importer un dashboard JSON
3. Personnaliser les requêtes PromQL
4. Configurer les alertes

### Ajouter des Alertes
1. Éditer `prometheus/alert_rules.yml`
2. Définir les conditions avec PromQL
3. Configurer les canaux de notification
4. Redémarrer Prometheus

---

## 🐛 Dépannage

### Problèmes Courants

**Prometheus ne scrape pas l'API**
```bash
# Vérifier l'endpoint
curl http://localhost:8000/metrics

# Vérifier la configuration
docker exec prometheus cat /etc/prometheus/prometheus.yml

# Vérifier les targets
curl http://localhost:9090/api/v1/targets
```

**Grafana ne se connecte pas à Prometheus**
```bash
# Vérifier la datasource
curl -u admin:admin123 http://localhost:3000/api/datasources

# Vérifier le réseau
docker network inspect docker_vacances-network
```

**OSRM Exporter ne fonctionne pas**
```bash
# Vérifier la connectivité OSRM
docker exec osrm-exporter curl http://osrm-car:5000/health

# Vérifier les logs
docker logs osrm-exporter
```

---

## 📚 Documentation Complémentaire

- **Prometheus** : https://prometheus.io/docs/
- **Grafana** : https://grafana.com/docs/
- **Prometheus Client** : https://github.com/prometheus/client_python
- **FastAPI** : https://fastapi.tiangolo.com/

---

*Monitoring TripMaNGo - Surveillance complète et proactive de votre infrastructure*
