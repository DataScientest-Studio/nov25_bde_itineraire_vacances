# Architecture – Itinéraire Vacances (Prime)

## Vue d’ensemble
Le projet **Itinéraire Vacances – Prime** repose sur une architecture data orientée recommandation et exploration géographique.

L’objectif est de proposer des itinéraires cohérents à partir de données touristiques hétérogènes, tout en garantissant performance, lisibilité et évolutivité.

---

## Architecture globale

- Sources de données
  - DataTourisme (POI, itinéraires, événements)
  - Sources complémentaires (Tripadvisor, Airbnb)
- ETL & Normalisation
- Stockage analytique (Parquet / PostGIS)
- Moteur Prime
  - Sélection des POI
  - Scoring & priorisation
  - Construction d’itinéraires
- Application Web
- yaml Copier le code

---

## Couche données

### Sources
- **DataTourisme** : source officielle principale
- Données open data complémentaires (Tripadvisor, Airbnb)

### Formats
- Ingestion : JSON (archive ZIP)
- Stockage analytique : Parquet
- Géospatial : PostGIS / H3 (selon besoins)

---

## ETL

### Étapes principales
1. Téléchargement du flux DataTourisme
2. Décompression des archives
3. Normalisation des POI
4. Filtrage fonctionnel (Prime)
5. Enrichissement géographique
6. Écriture Parquet

### Principes
- pipeline reproductible
- traçabilité des sources
- séparation données brutes / données traitées

---

## Moteur Prime (conceptuel)

Le moteur **Prime** repose sur :
- la cohérence géographique
- les préférences utilisateur (v1)
- la diversité des catégories (v1)
- la densité spatiale via des itinéraires (V2)

Les itinéraires touristiques sont traités comme des structures spatiales et non comme de simples POI.

---

## Scalabilité & évolutions

---

## Choix techniques clés
- Python (ETL)
- Parquet (analytique)
- PostGIS
- Application Web (Streamlit)
