# ETL DataTourisme webserice
Pipeline ETL pour extraire, transformer et charger les points d'intérêt (POI) de la région Île-de-France(IDF) depuis le flux Datatourisme vers une base de données PostgreSQL/PostGIS

Il est possible de remplacer le flux de données POI de la région IDF par un flux similaire (zip de fichiers JSON) sur un périmètre géographique différent.

La création du flux se réalise sur le site : https://diffuseur.datatourisme.fr/fr/login
La documentation de l'interface diffuseur : https://info.datatourisme.fr/2018/04/09/documentation-application-diffuseurs-datatourisme/
- à noter que le flux de données est disponible dans les 24h qui suivent sa création.

## Installation

### Prérequis
- Python 3.8+
- PostgreSQL 16 avec extension PostGIS
- Docker compose

### Setup local

```bash
# Créer un environnement virtuel
python -m venv project_env
source project_env/bin/activate  

# Installer les dépendances
pip install -r requirements.txt
```

### Variables d'environnement :

Créer un fichier `postgres_postgis/.env` :
```
DB_HOST=localhost
DB_PORT=5432 
DB_NAME=poi_db  # choisir un nom pour la base de données
DB_USER= admin  # choisir un nom d'utilisateur
DB_PASSWORD=votre_password # choisir un mot de passe
```
### Configuration Docker :

```bash
cd postgres_postgis
docker compose up -d
```
Cela lance une base PostgreSQL avec PostGIS sur `localhost:5432`.


## Utilisation

```bash
python3 main.py
```

Le script exécute le pipeline complet :
1. **Extraction** : récupère les POI du flux Datatourisme
2. **Transformation** : nettoie et structure les données
3. **Chargement** : insère les données dans PostgreSQL

## Structure du projet

```
etl_datatourisme_webservice/
├── etl/
│   ├── extract.py        # Extraction depuis l'API Datatourisme
│   ├── extract_opt.py    # Version optimisée de l'extraction avec parallélisation de la lecture des fichiers JSON de POI
│   ├── transform.py      # Transformation des données
│   ├── load.py           # Chargement dans PostgreSQL
│   ├── config/           # Fichiers de configuration
│   └── utils/            # Fonctions utilitaires
├── postgres_postgis/     # Configuration Docker PostgreSQL
├── main.py               # Point d'entrée du pipeline
├── requirements.txt      # Dépendances Python
└── POI_DB_MLD.drawio     # Modèle logique de données
```

## Technologies

- **Python** : langage principal
- **PostgreSQL/PostGIS** : base de données spatiales
- **Pandas** : manipulation de données
- **Psycopg2** : connecteur PostgreSQL
- **H3** : indexation spatiale hiérarchique
- **Docker** : conteneurisation