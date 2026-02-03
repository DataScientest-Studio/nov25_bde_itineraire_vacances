# Itinéraire Vacances — API Prime (Gold Contract)

## 1) Objectif

Cette API expose des **points d’intérêt (POI) classés** selon un score propriétaire *Prime*.  
Elle repose **exclusivement** sur des **vues PostgreSQL du schéma `gold`**, garantissant un **contrat stable** pour le frontend et les consommateurs API.

**Aucune requête directe vers `silver` ou `raw` n’est autorisée côté API.**

---

## 2) Architecture (Data → API)
raw → silver → gold → API / Frontend

Rôles des couches :
- raw : Données brutes (sources externes, fichiers, flux)
- silver : Logique métier, enrichissements, scoring, calculs intermédiaires. (Cette couche peut évoluer librement)
- gold : Contrat figé, stable, prêt pour exposition API (Lecture seule)
- API : Couche de lecture simple (read-only), sans logique métier

## 3) Source de vérité

### Vue principale API (v1)

La vue suivante constitue le **contrat API officiel v1** pour les POI Prime :
- Schéma : gold
- Vue : v_api_poi_prime
Toute consommation API doit passer par cette vue.

---

## 4) Schéma contractuel (v1)

|     Champ     |      Type |             Description                      |
|---------------|-----------|----------------------------------------------|
| `poi_id`      | string    | Identifiant stable du POI (URL DataTourisme) |
| `name`        | string    | Nom du point d’intérêt                       |
| `category`    | string    | Catégorie principale compressée              |
| `format`      | string    | Typologie de format                          |
| `tempo`       | string    | Typologie de tempo                           |
| `final_score` | float     | Score Prime final (gold)                     |
| `computed_at` | timestamp | Date de calcul du score                      |



### 5) Règles de stabilité
- Aucun champ **ne doit être supprimé ou renommé** en v1
- Tout changement incompatible ⇒ **nouvelle version**
  - ex : `gold.v_api_poi_prime_v2`
- Les ajouts de champs sont possibles uniquement s’ils sont non cassants
---


## 6) Requêtes SQL de référence

### Top POI Prime

SELECT *
FROM gold.v_api_poi_prime
ORDER BY final_score DESC
LIMIT 10;

### Pagination
Exemple de pagination simple :

SELECT *
FROM gold.v_api_poi_prime
ORDER BY final_score DESC
LIMIT 50 OFFSET 0;

### Filtrage simple
Filtrage par catégorie :

SELECT *
FROM gold.v_api_poi_prime
WHERE category = 'Patrimoine'
ORDER BY final_score DESC;


## 7) Endpoints API (exemple)
GET /v1/poi/prime
Paramètres de requête possibles :
- limit (entier, par défaut 50)
- offset (entier, par défaut 0)
- category (string, optionnel)
- format (string, optionnel)
- tempo (string, optionnel)


## 8) Exemple de réponse JSON
{
"meta": {
"limit": 2,
"offset": 0,
"count": 2
},
"data": [
{
"poi_id": "https://data.datatourisme.fr/42/462e6825-56eb-325f-8fe3-201d19bff87e
",
"name": "Le Monument aux Morts du Maquis des Manis",
"category": "Patrimoine",
"format": "Visite",
"tempo": "Court",
"final_score": 1.062,
"computed_at": "2026-01-26T18:33:59Z"
}
]
}


## 9) Vérification côté base

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'gold'
  AND table_name = 'v_api_poi_prime'
ORDER BY ordinal_position;


## 10) Principes de conception
- Gold = API contract
- Silver = libre d’évoluer
- Les vues gold sont :
  - déterministes
  - documentées
  - versionnées
  - indexées si nécessaire


## 11) Évolutions futures
- /v1/poi/prime/top
- /v1/poi/prime/by-category
- /v1/poi/prime/{poi_id}

Version v2 avec :
- géolocalisation
- score contextualisé
- personnalisation utilisateur


## 12) Licence & données
Les données sources (ex : DataTourisme) restent soumises à leurs licences respectives.
Ce projet n’expose que des vues dérivées et ne redistribue pas les datasets bruts.


## 13) Quality Gate (PRIME)
L’API PRIME intègre un mécanisme de contrôle qualité agissant comme un interrupteur de sécurité.
Il permet de détecter des incohérences de calcul avant exposition API.
- Mode STRICT : bloque la réponse si des erreurs critiques sont détectées
- Mode RELAXED : expose les résultats tout en remontant les alertes qualité

Ce mécanisme permet de sécuriser le contrat API tout en autorisant l’expérimentation.

