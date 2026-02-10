TripMaNGo
==============================

Avec TripMaNGo, ne planifiez plus vos voyages, profitez-en !
TripMaNGo pour "Trip Mapping Advendture & Guidance Optimizer". 


Sommaire
------------
* [À propos](#à-propos)
* [Architecture du Projet](#architecture-du-projet)
* [Démarrage Rapide](#démarrage-rapide)
* [Documentation Détaillée](#documentation-détaillée)


## À propos
On a tous vécu cette situation : dix onglets ouverts, des cartes, des blogs, des avis contradictoires… et au final, beaucoup de temps perdu pour un résultat souvent approximatif. C’est un vrai casse‑tête, et pourtant, organiser ses vacances devrait être un moment agréable. C’est ce constat qui a été le point de départ du projet.
Très vite, on s’est rendu compte que ce problème touche tout le monde :
* les familles qui veulent optimiser leur temps,
* les couples qui veulent profiter sans stress,
* les voyageurs solo qui veulent découvrir plus,
* et les touristes qui ne connaissent pas la région.
Notre objectif a été simple : aider chacun d’eux à vivre une expérience fluide et adaptée. 

* **Problème :** Mais derrière cette simplicité apparente, il y a une vraie complexité technique.
Les points d’intérêt sont nombreux et hétérogènes.
Les distances et durées dépendent du mode de transport.
Les contraintes utilisateur sont multiples.
Les solveurs ont des comportements différents.
Et l’itinéraire doit rester cohérent (nombre de restaurants proposés par jour), lisible et réaliste.
* **Solution :** Pour relever ce défi, nous avons construit un pipeline modulaire.
Chaque module joue un rôle précis :
La récupération des POI depuis la base de données,
Le filtrage à mutiples facettes (slustering, rebalancing, scoring ..) applique les préférences,
Le calcul des distances et durées via OSRM,
Le choix du solveur optimal,
L'assemblage de l'itinéraire,
L'enrichissement des données pour l'itinéraire afin de fournir un résultat clair et exploitable.
Cette architecture nous permet d’être robustes, évolutifs et transparents.

---
## Architecture de l'application

| Dossier | Composant | Rôle | Documentation |
| :--- | :--- | :--- | :--- |
| `app/` | **FastAPI** | API principale et logique métier | [Lire la Doc API](./app/README.md) |
| `dags/` | **Airflow** | Orchestration des pipelines de données | [Lire la Doc Data](./dags/README.md) |
| `reports/` | **Streamlit** | Dashboard de visualisation des données | [Lire la Doc Report](./reports/README.md) |
| `docker/` | **Ops** | Configuration des services (OSRM, DB, etc.) | - |
| `src/` | **Legacy** | Anciennes sources et utilitaires | - |

---

## Architecture du dossier Projet

```
├── app/                    # API FastAPI principale
├── src/                    # Sources legacy
├── docker/                 # Configuration Docker
│   ├── api/               # API Dockerisée
│   ├── osrm/              # Service OSRM
│   └── docker-compose.yml
├── data/                   # Données
├── dags/                   # Airflow DAGs
├── reports/               # Streamlit
├── tests/                 # Tests
├── README.md              # Documentation principale
└── docker-compose.yml     # Compose principal
```
---

## Services Dockerisés

### API FastAPI
- **Port** : 8000
- **Documentation** : http://localhost:8000/docs
- **Fonctions** : Catégories, POI, Itinéraires

### OSRM Multi-Profils
- **car** : http://localhost:5001
- **bike** : http://localhost:5002  
- **foot** : http://localhost:5003

### PostgreSQL + PostGIS
- **Port** : 5433
- **Base** : vacances

## Technologies

- **FastAPI** : Framework API
- **OSRM** : Routage
- **PostgreSQL** + **PostGIS** : BDD géospatiale
- **Polars** : DataFrames
- **H3** : Indexation géospatiale
- **Airflow** : ETL
- **Streamlit** : Dashboard


## Démarrage Rapide

### Pré-requis
* Docker & Docker Compose
* Un fichier `.env` à la racine (voir `.env.example`)

### Installation
Pour lancer l'intégralité de la stack en une commande :

```bash
# Cloner le projet
git clone https://github.com/ton-pseudo/nov25_bde_itineraire_vacances.git

# Entrer dans le dossier
cd nov25_bde_itineraire_vacances

# Lancer avec Docker
docker-compose up --build
```

### Services Individuels
```bash
# API seule
cd docker/api/
./deploy.sh

# OSRM seul  
cd docker/osrm/
./deploy.sh
```


## Contributing

1. Fork du projet
2. Branche feature : `git checkout -b feature/nouvelle-fonction`
3. Commit : `git commit -am 'Ajout nouvelle fonction'`
4. Push : `git push origin feature/nouvelle-fonction`
5. Pull Request


## Configuration

Le projet utilise des variables d'environnement pour fonctionner. Un fichier modèle est disponible à la racine.

### Setup des variables
1. Copiez le fichier d'exemple :
   ```bash
   cp .env.example .env

# API Configuration

# Database

## License
[License](./LICENSE)

