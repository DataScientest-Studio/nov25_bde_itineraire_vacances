# App d'Itinéraire de Vacances
# (user-first, data-first, orientée produit Prime, modulaire)

**Prime** est un moteur de recommandation d’itinéraires touristiques fondé sur une architecture data modulaire.  
Il combine des données touristiques ouvertes, des signaux analytiques tiers et un modèle de scoring pour proposer des parcours personnalisés (POI principaux, satellites, restaurants de midi).

---

## Architecture globale
![Prime architecture overview](sources/images/architecture_projet.png)
[Architecture du projet](architecture.md) : 
Le moteur Prime repose sur une architecture data modulaire allant de l’ingestion des données touristiques à la recommandation d’itinéraires (chaque brique de données a un rôle clair, indépendant, et remplaçable).

---

## Notebooks d’exploration
- [DataTourisme – Source de données](sources/datatourisme.md)
- [TripAdvisor – Signaux de popularité](sources/tripadvisor.md)
- [Airbnb – Hébergement (usage analytique)](sources/airbnb.md)
  
Ces notebooks documentent la compréhension des sources et les choix de modélisation.

---
## Pipelines de données (ETL)

Le dossier pipelines contient les pipelines Python responsables de l’ingestion, de la transformation et du chargement des données dans PostgreSQL.

### Extraction (Extract)
L’étape d’extraction alimente le moteur Prime à partir de sources hétérogènes, sans application de logique métier. Elle couvre :
- DataTourisme : ingestion via flux API JSON compressé, traitée en streaming pour gérer de gros volumes,
- TripAdvisor et Airbnb : lecture de fichiers CSV / Parquet utilisés comme signaux analytiques complémentaires.

Les scripts d’extraction assurent :
- une collecte fiable et reproductible,
- la traçabilité des sources et des dates d’ingestion,
- la production de données brutes conformes aux formats d’origine.
Les données extraites sont ensuite transmises aux étapes de transformation, puis chargées dans la couche Bronze, en amont du mode Prime.

### Ingestion (Load)
- [load_datatourisme_prime_classique.py](src/pipelines/load_datatourisme_prime_classique.py)
- [load_datatourisme_prime_experience.py](src/pipelines/load_datatourisme_prime_experience.py)
- [load_tripadvisor_france.py](src/pipelines/load_tripadvisor_france.py)
- [load_airbnb_paris.py](src/pipelines/load_airbnb_paris.py)

Ces scripts :
- consomment des flux API ou des fichiers Parquet,
- gèrent le parsing incrémental et le chargement contrôlé,
- alimentent la couche **Bronze**.

### Transformation (Transform)
- [transform_datatourisme_france.py](src/pipelines/transform_datatourisme_france.py)
- [transform_tripadvisor_france.py](src/pipelines/transform_tripadvisor_france.py)
- [transform_airbnb_paris.py](src/pipelines/transform_airbnb_paris.py)

Ces étapes :
- nettoient et normalisent les données,
- appliquent la logique métier (catégories, formats, tempo),
- construisent la couche **Silver** prête pour le scoring.

Les pipelines sont conçus pour être :
- idempotents,
- exécutables de manière indépendante,
- compatibles avec des stratégies d’upsert incrémental.

Le calcul des scores Prime est déclenché ultérieurement via des vues matérialisées dans la couche **Gold**.

---

## Architecture BDD (PostgreSQL / PostGIS)
![Architecture_BDD overview](sources/images/architecture_bdd.png)
Le moteur Prime repose sur une architecture Bronze / Silver / Gold :
- Bronze (Raw) : Ingestion des données brutes (JSON, Parquet, API), sans logique métier.
- Silver (Curated) : Nettoyage, normalisation et enrichissement métier des POI
(cat scores, formats, tempo, déduplication, identifiant canonique poi_id).
- Gold (Scores & Serving) : Calcul matérialisé des scores Prime et vues optimisées pour l’API et l’UI.

La base de données est organisée autour de vues et tables spécialisées :

### Tables & vues clés GOLD
- Gold.mv_poi_score : Materialized View contenant les scores calculés (final_score, reco_score).
- Gold.v_poi_scored : Vue pivot joignant Silver + scores Gold (dataset enrichi).
- Gold.v_score_by_category : Agrégats analytiques (KPIs par catégorie).
- Gold.v_api_poi_prime : Vue contractuelle exposée à l’API (API-only).
Le calcul du score est centralisé dans la couche data (Gold).
L’API consomme ces scores sans recalcul lourd.
Le score Prime est défini par la formule :

**final_score = main_cat_weight × (1 + format_weight + tempo_weight)**

- [001_create_tripadvisor_tables](migrations/001_create_tripadvisor_tables.sql)
- [002_create_prime_tables](migrations/002_create_prime_tables.sql)
- [003_fix_prime_postal_code_text](migrations/003_fix_prime_postal_code_text.sql)
- [004_create_gold_poi_score](migrations/004_create_gold_poi_score.sql)
- [005_create_gold_views](migrations/005_create_gold_views.sql)
- [006_create_gold_api_views](migrations/006_create_gold_api_views.sql)

---

## API interne (serving layer)
L’API interne [openapi.yml](api/openapi.yml) assure :
- la lecture des données depuis PostgreSQL (vues Gold),
- l’application des filtres Prime (zones, catégories, contraintes),
- le ranking des POI (ORDER BY score),
- l’identification des POI satellites et des restaurants de midi,
- l’enrichissement via le temps de marche (OSRM).
Exemples de routes : /zones --> /poi --> /prime (ranking + filters) --> /walk-time

Cette spécification sert à :
- garantir la stabilité du contrat entre l’API et l’UI,
- faciliter la compréhension du fonctionnement de l’API,
- permettre la génération automatique de documentation ou de clients.
Le scoring n’est pas calculé dans l’API : les endpoints consomment les scores pré-calculés dans la couche **Gold**.

---
## Quality Gate du scoring PRIME

Le score **PRIME** est un score critique exposé à l’API et au frontend.
Afin d’éviter toute exposition de résultats incohérents (bug de calcul, données corrompues, régression),
une **Quality Gate explicite** est placée entre le calcul du score et sa publication.

### Principe

La Quality Gate agit comme un **interrupteur de sécurité** :

- 🟢 **RELAXED** : le score est exposé, mais les anomalies sont détectées, comptabilisées et journalisées
- 🔴 **STRICT** : toute anomalie bloquante empêche l’exposition du score PRIME

Ce mécanisme permet :
- de détecter immédiatement un faux calcul PRIME
- de sécuriser les démonstrations et environnements de production
- de conserver un mode souple pour le prototypage (MVP)

Le mode est contrôlé par variable d’environnement :

**PRIME_QUALITY_MODE=STRICT | RELAXED**

### Rôle des composants
- ops/data_checks/ : règles de validation métier du score PRIME
- ops/quality/ : orchestration de la Quality Gate (blocage / autorisation)
- domain/prime.py : calcul du score PRIME (source du risque)

La Quality Gate est volontairement séparée du calcul afin de pouvoir détecter
les erreurs même lorsque le calcul lui-même est incorrect.

- [Règles de contrôle PRIME](src/ops/data_checks/prime_checks.py)
- [Quality Gate (STRICT / RELAXED)](src/ops/quality/quality_gate.py)
- [Calcul du score PRIME](src/domain/prime.py)
---
## Stack Docker (optionnelle)

Une stack Docker optionnelle est disponible pour le développement local et les tests.
Elle comprend notamment :
- Postgres / PostGIS
- Un backend FastAPI
- Une interface Streamlit
- Des jobs ETL batch

L’ensemble des fichiers liés à Docker est isolé dans le dossier [thuvo-docs/docker/]
- Docker Compose: [docker-compose.yml](docker/docker-compose.yml)
- Environment template: [.env.example](docker/.env.example)

Les modalités d’utilisation et les détails techniques sont documentés localement dans ce répertoire.
---

## Note légale

Ce projet utilise des données issues de sources ouvertes et de sources tierces à des fins d’analyse, de démonstration et de recherche.

- **DataTourisme** est utilisé comme source officielle de points d’intérêt touristiques en open data.
- **TripAdvisor** et **Airbnb** sont utilisés uniquement comme sources analytiques indirectes (signaux agrégés, statistiques, densité), sans redistribution de données propriétaires.

Aucune donnée brute, aucun contenu propriétaire (textes, images, liens, identifiants publics) issu de TripAdvisor ou Airbnb n’est stocké ni exposé à l’utilisateur final.

Ce projet n’est ni affilié, ni soutenu par TripAdvisor ou Airbnb.

---
## Structure du dépôt

