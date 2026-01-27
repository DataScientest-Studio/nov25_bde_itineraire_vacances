# DataTourisme – Source de données officielle

## Présentation
DataTourisme est la plateforme nationale française de diffusion des données d’information touristique en open data.

Dans le cadre du projet **Itinéraire Vacances – Prime**, DataTourisme constitue la **source principale de points d’intérêt (POI)** :
- lieux culturels et naturels
- activités et loisirs
- hébergements et restauration
- itinéraires touristiques (pédestre, cyclable, fluvial)
- événements culturels et sportifs

---

## Périmètre fonctionnel (choix Prime)

### Couverture géographique : France métropolitaine

> Le périmètre métropolitain garantit une cohérence géographique, des distances comparables et une intégration homogène avec les moteurs de routage (OSRM).

---

### Types de données sélectionnées
Le mode **Prime** privilégie les POI à forte valeur expérientielle.

**Inclus :**
- Sites naturels
- Sites culturels
- Loisirs et activités
- Fournisseurs de dégustation
- Restauration
- Hébergement
- Produits touristiques (visite, pratique, location)
- Itinéraires :
  - pédestres
  - cyclables
  - fluviaux / maritimes
- Événements culturels et sports/loisirs

**Exclus (MVP Prime) :**
- services pratiques
- transport
- santé
- événements professionnels/commerciaux
- itinéraires routiers, équestres, sous-marins

---

## Mode d’accès aux données

### Type d’accès
- API officielle DataTourisme
- Flux généré par la plateforme (batch planifié)
- Authentification par clé API

### Format choisi
- **JSON (archive ZIP)**

Le flux génère :
- un fichier d’index (liste des POI + date de modification)
- un dossier `objects/` contenant un fichier JSON par POI
- un fichier de contexte (JSON-LD)

Ce format permet :
- un parsing simple en Python
- une traçabilité des sources
- une conversion directe vers Parquet
- une évolution vers un pipeline incrémental (V2)

---

## Rafraîchissement des données
- Génération planifiée quotidienne (batch nocturne 23h)
- Les données sont considérées comme **quasi temps réel à l’échelle touristique**

---

## Pipeline d’ingestion (résumé)

DataTourisme API (ZIP JSON) --> Décompression --> Normalisation JSON --> Filtrage Prime --> Parquet (France entière) --> yaml Copier le code

---

## Schéma de données (MVP)

### Table `poi.parquet`
- `poi_id` : identifiant stable
- `name` : nom du POI
- `lat`, `lon` : coordonnées géographiques
- `prime_bucket` : catégorie fonctionnelle Prime
- `city`, `postal_code`, `insee`, `department`, `region`
- `raw_file` : traçabilité source

### Table `poi_types.parquet`
- `poi_id`
- `type` (ontologie DataTourisme / RDF)

---

## Rôle dans le moteur Prime

Les données DataTourisme alimentent :
- la sélection des POI candidats
- la construction d’itinéraires cohérents
- la logique de priorisation Prime (What/where/why/when/how)
- le calcul de densité et de diversité via itinéraires (pédestres, cyclables, fluviaux / maritimes)

Les itinéraires touristiques sont traités comme des **structures spatiales** et non comme de simples POI.

---

## Limites connues
- hétérogénéité des champs selon les territoires (par exemple : address structuré ou non structuré : rue, code postal, commune. openingHours partiellement renseignés...)
- qualité variable des catégories sémantiques :
  - Un même lieu peut être catégorisé "Parc de loisir" et "Activité familiale"
  - Certains POI ont plusieurs types très génériques "site religieuse", d’autres très précis mais peu exploitable "événement regligieuse"
- géométries parfois incomplètes : absence latitude, longitude

Ces limites sont gérées par :
- normalisation
- filtrage Prime
- règles de fallback dans l’algorithme

---

## Licence et conformité
Les données sont issues de DataTourisme,
plateforme nationale d’open data touristique.

L’application ne redistribue pas les données brutes,
mais les exploite dans un cadre de recommandation et de visualisation.
