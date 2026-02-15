# ItineraryPipeline - Pipeline d'Optimisation d'Itinéraires

Le `ItineraryPipeline` est le cœur du système de calcul d'itinéraires optimisés. Il orchestre l'ensemble du processus depuis les points d'intérêt bruts jusqu'à l'itinéraire final enrichi.

## Architecture du Pipeline

Le pipeline suit 5 étapes séquentielles :

```
POIs Bruts → Clustering Spatial → Préparation OSRM → Matrices OSRM → Solveur → Enrichissement
```

## Structure de la Classe

```python
class ItineraryPipeline:
    """
    Pipeline complet :
        1. Clustering spatial
        2. Préparation OSRM
        3. OSRM matrices
        4. Solveur (NN2O / GA / AUTO)
        5. Enrichissement
    """
```

---

## Étape 1 - Clustering Spatial

### Méthode : `_cluster_pois()`

**Objectif** : Regrouper les POIs en clusters géographiques par jour

**Signature** :
```python
def _cluster_pois(self, pois_df: pl.DataFrame, nb_days, anchor_lat, anchor_lon) -> pl.DataFrame
```

**Paramètres** :
- `pois_df` : DataFrame des POIs avec colonnes `latitude`, `longitude`, `score`
- `nb_days` : Nombre de jours du voyage
- `anchor_lat`, `anchor_lon` : Point de référence (hôtel, centre-ville)

**Résultat** : DataFrame avec colonnes ajoutées :
- `cluster_id` : Identifiant du cluster (0 à nb_days-1)
- `h3_r9` : Index géospatial H3 pour optimisation

**Détails d'implémentation** :
```python
return (
    SpatialClusterer(pois_df.lazy())
    .set_nb_days(nb_days)
    .set_anchor(anchor_lat, anchor_lon)
    .apply()
    .collect()
)
```

---

##  Étape 2 - Préparation OSRM

### Méthode : `_build_osrm_ready_pois()`

**Objectif** : Formater les POIs pour les requêtes OSRM et limiter la complexité

**Signature** :
```python
def _build_osrm_ready_pois(
    self,
    df_prepared: pl.DataFrame,
    mode: str = "walk",
    max_pois_per_cluster: int = 50,
    min_score: float = 0.2,
) -> pl.DataFrame
```

**Paramètres** :
- `mode` : Mode de transport (`walk`, `bike`, `car`)
- `max_pois_per_cluster` : Limite pour éviter l'explosion combinatoire
- `min_score` : Score minimum pour garder un POI

**Transformations** :
1. **Renommage** : `day` → `cluster_id` si nécessaire
2. **Filtrage** : Suppression des POIs sous le score minimum
3. **Limitation** : Maximum de POIs par cluster
4. **Indexation** : Ajout de `osrm_index` pour les matrices

**Résultat** : DataFrame prêt pour OSRM avec :
- `cluster_id` : Jour du voyage
- `osrm_index` : Index global pour matrices
- POIs filtrés et limités

---

## Étape 3 - Post-Clustering et Rééquilibrage

### Module : `post_clustering.py`

**Objectif** : Optimiser la distribution des POIs après clustering pour garantir des itinéraires équilibrés et réalistes

Cette étape cruciale intervient entre le clustering spatial et la préparation OSRM pour affiner la sélection des POIs.

### Fonctionnalités Principales

#### 1. Rééquilibrage Intelligent des Restaurants

**Problème** : Sur-représentation des restaurants qui déséquilibre les itinéraires

**Solution** : `smart_restaurant_sampling()`
```python
def smart_restaurant_sampling(df: pl.DataFrame, max_per_subcat_per_cell: int = 2) -> pl.DataFrame:
    # Limite les restaurants par sous-catégorie et par cellule H3
    restos = (
        df.filter(pl.col("sub_category").is_in(RESTAURANT_SUBCATEGORIES))
          .sort("final_score", descending=True)
          .group_by(["sub_category", "h3_r7"])
          .head(max_per_subcat_per_cell)
    )
```

**Catégories gérées** :
- `Restaurants`
- `Restauration rapide` 
- `Bars & cafés`

#### 2. Diversification par Catégorie

**Objectif** : Éviter la sur-représentation d'une seule catégorie

**Fonction** : `ensure_minimum_per_category()`
```python
def ensure_minimum_per_category(df: pl.DataFrame, max_per_category: int = 10) -> pl.DataFrame:
    # Limite par catégorie principale pour garantir la diversité
    grouped = (
        non_resto.sort("final_score", descending=True)
                 .group_by("main_category")
                 .head(max_per_category)
    )
```

#### 3. Contrôle de Densité

**But** : Éviter la concentration excessive de POIs dans une même zone

**Fonction** : `limit_density()`
```python
def limit_density(df: pl.DataFrame, max_per_cell: int = 10) -> pl.DataFrame:
    # Limite le nombre de POIs par cellule H3 (résolution 7)
    limited = (
        df.sort("final_score", descending=True)
          .group_by("h3_r7")
          .head(max_per_cell)
    )
```

#### 4. Filtrage par Mode de Transport

**Logique** : Adapter les POIs aux contraintes de déplacement

**Fonction** : `filter_by_transport_mode()`
```python
def filter_by_transport_mode(df: pl.DataFrame, mode: TransportMode) -> pl.DataFrame:
    # Rayons max selon mode de transport
    TRANSPORT_MAX_RADIUS_KM = {
        "walk": 14.0,   # Marche : rayon modéré
        "bike": 27.0,   # Vélo : rayon étendu  
        "car": 40.0,    # Voiture : grand rayon
    }
```

**Processus** :
1. **Calcul centroïde** : Point central de chaque cluster
2. **Distance POI-centroïde** : Formule Haversine
3. **Filtrage** : POIs dans le rayon compatible

### Pipeline Complet de Post-Clustering

**Fonction principale** : `build_osrm_ready_pois()`

**Étapes séquentielles** :

1. **Rééquilibrage initial**
   ```python
   df = rebalance_pois(df)  # Application des 3 stratégies
   ```

2. **Split restaurants**
   ```python
   df_filtered = split_restaurants_and_others(df, k_restos=3)
   ```

3. **Filtrage transport**
   ```python
   df_transport_filtered = filter_by_transport_mode(df_filtered, mode=mode)
   ```

4. **Filtrage par score**
   ```python
   df_score_filtered = filter_by_final_score(
       df_transport_filtered,
       max_pois_per_cluster=max_pois_per_cluster,
       min_score=min_score
   )
   ```

5. **Préparation OSRM**
   ```python
   df_osrm = prepare_osrm_nodes(df_score_filtered)
   ```

### Métriques de Qualité

#### Score de Diversité
```python
# Combinaison score original + diversité géographique
score_diversity = final_score * 0.2 + diversity_commune_norm * 0.8
```

#### Logs de Suivi
```python
logger.info(f"[rebalance] initial POIs: {df.height}")
logger.info(f"[rebalance] restos kept: {restos.height}, others: {others.height}")
logger.info(f"[rebalance] final POIs: {df3.height}")
```

### Impact sur les Itinéraires

**Avantages** :
- **Équilibre nutritionnel** : Pas trop de restaurants
- **Diversité culturelle** : Toutes les catégories représentées
- **Réalisme** : Distances adaptées au transport
- **Performance** : Réduction complexité algorithmique

**Exemple de transformation** :
```
Avant post-clustering :
- 50 POIs total
- 30 restaurants (60%)
- 20 autres catégories

Après post-clustering :
- 25 POIs total  
- 6 restaurants (24%)
- 19 autres catégories (76%)
```

---

## Étape 4 - Matrices OSRM

### Méthodes : `_compute_osrm_matrices()` et `_get_osrm_profile()`

**Objectif** : Calculer les matrices de distances et durées entre tous les POIs

**Signature** :
```python
async def _compute_osrm_matrices(
    self,
    df_clustered: pl.DataFrame,
    osrm: OSRMClientAsync,
    transport_mode: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
```

**Profils OSRM** :
```python
def _get_osrm_profile(self, transport_mode: str) -> str:
    if transport_mode == "walk": return "foot"
    if transport_mode == "bike": return "bike"
    if transport_mode == "car": return "driving"
    return "foot"
```

**Processus asynchrone** :
1. **Préparation** : Formatage des coordonnées pour OSRM
2. **Appels parallèles** : Requêtes OSRM optimisées
3. **Construction** : Matrices carrées distances/durées

**Retour** :
- `df_clustered` : DataFrame avec indices OSRM
- `df_osrm_dist` : Matrice des distances (mètres)
- `df_osrm_dur` : Matrice des durées (secondes)

---

## Étape 5 - Solveurs d'Optimisation

Le pipeline intègre 3 solveurs avec sélection automatique.

### Solveur NN2O (Nearest Neighbor 2-Opt)

**Méthode** : `_compute_itinerary_nn2o()`

**Caractéristiques** :
- **Rapidité** : Très rapide (< 100ms pour 50 POIs)
- **Qualité** : Bonne pour petits clusters
- **Idéal** : ≤ 6 POIs par jour

**Algorithme** :
1. **Extraction** : Matrice locale du cluster
2. **NN2O** : Nearest Neighbor + optimisation 2-Opt
3. **Remapping** : Indices locaux → globaux
4. **Construction** : DataFrame ordonné

**Code clé** :
```python
nn2o = NN2OptAlgo(poi_df=df_day_pd, duration_matrix=local_matrix)
best_route_local, best_cost = nn2o.run_nn2opt(try_all_starts=False)
```

### Solveur Génétique (GA)

**Méthode** : `_compute_itinerary_ga()`

**Caractéristiques** :
- **Qualité** : Excellente pour grands clusters
- **Performance** : Plus lent (1-5s pour 50 POIs)
- **Idéal** : > 6 POIs par jour

**Paramètres GA** :
```python
ga.setup_toolbox(itin_min_poi=5, itin_max_poi=15)
best_route_local, fitness = ga.run_ga(
    pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3
)
```

**Algorithme** :
1. **Population initiale** : Solutions aléatoires
2. **Évolution** : Sélection, croisement, mutation
3. **Optimisation** : Meilleure solution sur N générations
4. **Validation** : Vérification des indices

### Mode AUTO

**Logique intelligente** :
```python
if cluster_size <= 6:  # Seuil déterminé par benchmark
    chosen = "nn2o"
else:
    chosen = "ga"
```

**Avantages** :
- **Adaptatif** : Choisit le meilleur solveur par cluster
- **Optimal** : Meilleur compromis performance/qualité
- **Transparent** : `solver_used` indique le choix

---

## Étape 6 - Enrichissement

### Méthode : `enrich_itinerary()`

**Objectif** : Ajouter les métriques détaillées de l'itinéraire

**Signature** :
```python
def enrich_itinerary(
    self, 
    df_day, 
    matrix_durations, 
    matrix_distances, 
    order
) -> pl.DataFrame
```

**Métriques calculées** :

#### Distances
- `distance_from_prev_km` : Distance depuis POI précédent
- `cumulative_distance_km` : Distance cumulée du jour
- `day_total_distance_km` : Distance totale du jour

#### Durées
- `duration_from_prev_min` : Durée depuis POI précédent
- `cumulative_duration_min` : Durée cumulée du jour
- `day_total_duration_min` : Durée totale du jour

**Calculs** :
```python
# Pour chaque segment de l'itinéraire
for i in range(n - 1):
    d = float(matrix_distances[order[i], order[i + 1]])
    t = float(matrix_durations[order[i], order[i + 1]])
    distance_from_prev.append(d)
    duration_from_prev.append(t)

# Cumuls
cum_d += d
cum_t += t
cumulative_distance.append(cum_d)
cumulative_duration.append(cum_t)
```

---

## Pipeline Complet

### Méthode principale : `run_from_pois_df()`

**Signature** :
```python
async def run_from_pois_df(
    self,
    pois_df: pl.DataFrame,
    nb_days: int,
    anchor_lat: float,
    anchor_lon: float,
    osrm: OSRMClientAsync,
    transport_mode: str = "walk",
    solver: str = "nn2o",
    max_pois_per_cluster: int = 50,
    osrm_min_score: float = 0.2,
):
```

**Flux d'exécution** :

1. **Clustering spatial**
   ```python
   df_prepared = self._cluster_pois(pois_df, nb_days, anchor_lat, anchor_lon)
   ```

2. **Préparation OSRM**
   ```python
   df_clustered = self._build_osrm_ready_pois(df_prepared, mode=transport_mode, ...)
   ```

3. **Matrices OSRM**
   ```python
   df_clustered, df_osrm_dist, df_osrm_dur = await self._compute_osrm_matrices(...)
   ```

4. **Solveur**
   ```python
   if solver == "nn2o":
       optimizer, df_itinerary = self._compute_itinerary_nn2o(df_clustered, df_osrm_dur)
   elif solver == "ga":
       optimizer, df_itinerary = self._compute_itinerary_ga(df_clustered, df_osrm_dur)
   elif solver == "auto":
       # Choix intelligent par cluster
   ```

5. **Enrichissement**
   ```python
   df_enriched = self.enrich_itinerary(df_day, local_matrix_dur, local_matrix_dist, order)
   ```

**Retour** :
```python
return df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer
```

---

## Performance et Optimisations

### Complexité Algorithmique

| Étape | Complexité | Optimisations |
|-------|------------|--------------|
| Clustering | O(n log n) | Index H3, partitionnement |
| OSRM | O(n²) | Requêtes parallèles, cache |
| NN2O | O(n²) | Heuristique rapide |
| GA | O(g × p × n) | Paramètres adaptatifs |
| Enrichissement | O(n) | Calculs vectorisés |

### Seuils de Performance

**Basé sur benchmarks** ([voir benchmark README](../../src/benchmark/README.md)) :

| Taille cluster | Solveur recommandé | Temps moyen | Qualité |
|----------------|-------------------|-------------|---------|
| 2-6 POIs | NN2O | < 50ms | 🟊🟊🟊 |
| 7-20 POIs | GA | 200-800ms | 🟊🟊🟊🟊 |
| 21-50 POIs | GA | 1-3s | 🟊🟊🟊🟊🟊 |


---

## Utilisation Pratique

### Initialisation

```python
from app.pipeline.itinerary_pipeline import ItineraryPipeline
from app.pipeline.features.osrm import OSRMClientAsync

# Initialisation
pipeline = ItineraryPipeline()
osrm = OSRMClientAsync("http://localhost:5000")
```

### Exécution Simple

```python
# Données d'entrée
pois_df = pl.read_parquet("data/pois.parquet")

# Exécution du pipeline
df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer = await pipeline.run_from_pois_df(
    pois_df=pois_df,
    nb_days=5,
    anchor_lat=48.8566,
    anchor_lon=2.3522,
    osrm=osrm,
    transport_mode="walk",
    solver="auto"
)
```

### Résultats

**DataFrame final `df_itinerary`** :
```
┌───────────┬──────────────┬─────────────┬──────────────┬─────────────────┐
│ poi_id    │ cluster_id   │ order       │ solver_used  │ distance_from_ │
│ ---       │ ---          │ ---         │ ---          │ prev_km        │
│ i64       │ i64          │ i64         │ str          │ f64            │
├───────────┼──────────────┼─────────────┼──────────────┼─────────────────┤
│ 123       │ 0            │ 0           │ nn2o         │ 0.0            │
│ 456       │ 0            │ 1           │ nn2o         │ 0.8            │
│ 789       │ 0            │ 2           │ nn2o         │ 1.2            │
└───────────┴──────────────┴─────────────┴──────────────┴─────────────────┘
```

**Colonnes disponibles** :
- **Identifiants** : `poi_id`, `cluster_id`, `order`
- **Métadonnées** : `solver_used`, `osrm_index`
- **POI** : `nom_du_poi`, `latitude`, `longitude`
- **Distances** : `distance_from_prev_km`, `cumulative_distance_km`, `day_total_distance_km`
- **Durées** : `duration_from_prev_min`, `cumulative_duration_min`, `day_total_duration_min`

---

## Cas d'Usage Avancés

### Mode AUTO Intelligent

```python
# Le mode AUTO choisit le meilleur solveur par cluster
results = await pipeline.run_from_pois_df(
    pois_df=pois_df,
    nb_days=3,
    anchor_lat=48.8566,
    anchor_lon=2.3522,
    osrm=osrm,
    solver="auto"  # ← Choix intelligent
)


```

### Multi-transport

```python
# Comparaison des modes de transport
modes = ["walk", "bike", "car"]
results = {}

for mode in modes:
    _, _, _, df_itin, _ = await pipeline.run_from_pois_df(
        pois_df=pois_df,
        nb_days=5,
        anchor_lat=48.8566,
        anchor_lon=2.3522,
        osrm=osrm,
        transport_mode=mode,
        solver="auto"
    )
    results[mode] = df_itin["day_total_distance_km"].sum()
```

---

## Dépannage et Debug

### Logs Utiles

```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Points de contrôle dans le pipeline
print(f"POIs bruts: {pois_df.height}")
print(f"Clusters créés: {df_prepared['cluster_id'].n_unique()}")
print(f"POIs après filtrage: {df_clustered.height}")
print(f"Solveur utilisé: {optimizer}")
```

### Problèmes Courants

**OSRM timeouts** :
```python
# Vérifier la connexion OSRM
import aiohttp
async with aiohttp.ClientSession() as session:
    async with session.get("http://localhost:5000/route/v1/walk/2.3,48.8;2.4,48.9") as resp:
        print(resp.status)
```

**Clusters vides** :
```python
# Vérifier la distribution des tailles
cluster_sizes = df_prepared.groupby("cluster_id").count().sort("cluster_id")
print(cluster_sizes)
```

**GA sans solution** :
```python
# Augmenter les paramètres GA
ga.run_ga(pop_size=100, ngen=100, cxpb=0.8, mutpb=0.2)
```

---

## Références

**Modules connexes** :
- [SpatialClusterer](features/spatial_clustering.py) : Clustering géographique
- [OSRMClientAsync](features/osrm.py) : Client OSRM asynchrone
- [NN2OptAlgo](features/optimizer_nn2o.py) : Solveur NN2O
- [GeneticAlgo](features/optimizer_ga.py) : Algorithme génétique
- [Benchmark](../../src/benchmark/README.md) : Analyse de performance

**Documentation externe** :
- [OSRM Documentation](http://project-osrm.org/docs/v5.24.0/api/)
- [H3 Indexation](https://uber.github.io/h3/)

---

[Retour sur la documentation de l'API](../README.md)