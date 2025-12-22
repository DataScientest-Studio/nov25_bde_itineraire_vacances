# Objectif du projet
Ce projet vise à concevoir une couche de données géographiques structurée et exploitable permettant d’alimenter des services d’aide à la décision pour l’organisation de séjours touristiques (choix d’hébergement, sites touristiques, restauration, mobilité).

L’approche adoptée orientée Data Engineering, avec un focus sur :
- Batch pipeline : ingestion → normalisation → features → artefacts
- Serving : app Streamlit consomme les features / modèle
- Reproductibilité : Parquet + scripts src/ + requirements

# Architecture logique du pipeline
Sources de données :
→ Ingestion & nettoyage
→ Normalisation géographique
→ Dataset enrichi (layer analytics)
→ Consommation (visualisation / exploration)

# Sources de données utilisées

## 1. Sites touristiques (open data)
- Jeu de données : Principaux sites touristiques en Île-de-France (data.gouv)
- Format : JSON / CSV
- Contenu : localisation géographique, type de site touristique, informations administratives (commune, département)
- Link : https://data.iledefrance.fr/explore/dataset/principaux-sites-touristiques-en-ile-de-france0/api/

## 2. Hébergement
- Source : Airbnb (Paris / Île-de-France)
- Objectif : représenter les points d’ancrage potentiels des séjours
- Link : https://www.kaggle.com/datasets/juliatb/airbnb-paris?select=reviews.csv

## 3. Restauration
- Source : TripAdvisor (restaurants européens)
- Objectif : enrichissement de l’offre touristique autour des zones visitées
- Link : https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants


# Traitements Data Engineering réalisés
- Ingestion & nettoyage
- lecture de sources hétérogènes (formats variés)
- gestion des valeurs manquantes
- sélection des attributs pertinents pour un usage géographique
- Normalisation géographique
- extraction explicite des champs Latitude et Longitude
- harmonisation des coordonnées pour un usage cartographique
- validation des points géographiques exploitables
- Structuration des données
- constitution de DataFrames propres et cohérents
- séparation logique des datasets par type d’entité : hébergement, sites touristiques, restauration

# Couche de consommation (Serving / Analytics Layer)
Une carte interactive développée à partir de la couche de données nettoyée afin de valider :
- la qualité des données géographiques,
- la densité spatiale des POI,
- la pertinence métier des regroupements.

Fonctionnalités :
- clustering automatique des points (MarkerCluster)
- visualisation multi-échelle (zoom / exploration)
- affichage contextuel des informations clés (nom, type, commune)

Cette visualisation constitue un consumer de la couche de données, illustrant comment celle-ci peut être exploitée par :
- une application utilisateur,
- un service d’aide à la décision,
- ou un futur moteur d’optimisation d’itinéraires.

# Intégration future
La couche de données produite est conçue pour être directement consommable par :
- des API de routage (OSRM, OpenRouteService, etc.)
- des algorithmes de recommandation (sélection de POI)
- des services d’optimisation multi-critères (temps, distance, budget)

# Technologies utilisées
- Python
- Pandas / NumPy
- Folium
- Jupyter Notebook
- Données open data
