# Module Benchmark

Ce module permet d'évaluer et comparer les performances des solveurs d'optimisation d'itinéraires (GA vs NN2O) à travers différentes métriques : performance, stabilité et robustesse.

## Architecture du Module

```
src/benchmark/
├── README.md                    # Documentation (ce fichier)
├── benchmark_runner.py          # Orchestrateur principal
├── evaluate_solver.py           # Évaluation unitaire d'un solveur
├── compare_sizes.py             # Comparaison par tailles (seuil AUTO)
├── compare_stability.py         # Test de stabilité sur matrices
├── benchmark_ga_tuning.py      # Tuning des configurations GA
├── benchmark_io.py             # Sauvegarde/chargement des résultats
├── visualize_benchmark.py      # Visualisations des résultats
└── ../run_benchmark.py          # Script d'exécution complet
```

## Fichiers et Fonctionnalités

### 1. `benchmark_runner.py` - Orchestrateur Principal

**Rôle** : Point d'entrée unique qui coordonne tous les benchmarks

**Fonction principale** :
```python
run_benchmark(
    df_all_pois: pl.DataFrame,
    matrices: Dict[str, Dict[str, np.ndarray]],
    compute_ga, compute_nn2o, pipeline,
    cluster_sizes: List[int] = [2, 10, 25, 50],
    runs_per_matrix: int = 5,
    runs_per_size: int = 10
) -> Dict[str, pl.DataFrame]
```

**Retourne** :
- `size_comparison` : Comparaison GA vs NN2O par tailles
- `stability_ga` : Stabilité du solveur GA
- `stability_nn2o` : Stabilité du solveur NN2O

**Utilisation** :
```python
results = run_benchmark(
    df_all_pois=df_clustered,
    matrices=matrices,
    compute_ga=pipeline._compute_itinerary_ga,
    compute_nn2o=pipeline._compute_itinerary_nn2o,
    pipeline=pipeline
)
```

---

### 2. `evaluate_solver.py` - Évaluation Unitaire

**Rôle** : Évalue un solveur sur un cluster spécifique avec métriques détaillées

**Fonction principale** :
```python
evaluate_solver_on_cluster(
    solver: Literal["ga", "nn2o"],
    df_cluster: pl.DataFrame,
    matrix_dur, matrix_dist: np.ndarray,
    compute_ga, compute_nn2o, pipeline,
    matrix_id: str, run_id: int
) -> Dict[str, Any]
```

**Métriques retournées** :
- `runtime_ms` : Temps d'exécution en millisecondes
- `total_distance` : Distance totale de l'itinéraire
- `total_duration` : Durée totale de l'itinéraire  
- `order_signature` : Hash MD5 de l'ordre (pour stabilité)
- Gestion des échecs (valeurs `None` si solveur échoue)

**Utilisation** : Appelée par les modules de comparaison, pas directement

---

### 3. `compare_sizes.py` - Seuil AUTO

**Rôle** : Détermine le seuil optimal pour basculer entre NN2O et GA

**Fonction principale** :
```python
compare_solvers_on_sizes(
    df_all_pois: pl.DataFrame,
    matrix_dur, matrix_dist: np.ndarray,
    compute_ga, compute_nn2o, pipeline,
    sizes: List[int],
    runs_per_size: int = 5,
    matrix_id: str = "walk"
) -> pl.DataFrame
```

**Objectif** : Identifier à partir de quelle taille le GA devient plus performant que NN2O

**Résultat typique** :
```
cluster_size | solver | runtime_ms | total_distance
     2       |  nn2o  |    15      |     1200
     2       |   ga   |    150     |     1150
    10       |  nn2o  |    45      |     3400
    10       |   ga   |    280     |     3100
```

---

### 4. `compare_stability.py` - Test de Robustesse

**Rôle** : Évalue la stabilité des solveurs sur différentes matrices OSRM

**Fonction principale** :
```python
compare_stability_on_matrices(
    df_all_pois: pl.DataFrame,
    matrices: Dict[str, Dict[str, np.ndarray]],
    compute_ga, compute_nn2o, pipeline,
    solver: str,
    cluster_sizes: List[int] = [2, 10, 25, 50],
    runs_per_matrix: int = 3
) -> pl.DataFrame
```

**Matrices testées** :
- `walk` : Mode piéton normal
- `bike` : Mode vélo
- `car` : Mode voiture
- `walk_perturbed` : Mode piéton avec perturbation (+2%)

**Métriques de stabilité** :
- Variance des distances/durées
- Consistance des ordres (order_signature)
- Robustesse aux changements de transport

---

### 5. `benchmark_ga_tuning.py` - Optimisation GA

**Rôle** : Teste différentes configurations de l'algorithme génétique

**Configurations prédéfinies** :
```python
GA_CONFIGS = {
    "fast": {"pop_size": 20, "ngen": 40, "cxpb": 0.9, "mutpb": 0.05},
    "balanced": {"pop_size": 30, "ngen": 60, "cxpb": 0.8, "mutpb": 0.10},
    "premium": {"pop_size": 40, "ngen": 80, "cxpb": 0.8, "mutpb": 0.10},
}
```

**Fonction principale** :
```python
run_ga_tuning(
    df_clustered: pl.DataFrame,
    matrices: dict,
    matrix_id: str = "walk",
    runs_per_cluster: int = 3
) -> pl.DataFrame
```

**Objectif** : Trouver le meilleur compromis performance/qualité

---

### 6. `benchmark_io.py` - Sauvergarde / Chargement

**Rôle** : Sauvegarde et chargement des résultats de benchmark

**Fonctions** :
```python
save_benchmark(results: Dict[str, pl.DataFrame], folder: str = "benchmark_results")
load_benchmark(folder: str = "benchmark_results") -> Dict[str, pl.DataFrame]
```

**Format** : Fichiers Parquet (rapide, compact, compatible Polars)

**Utilisation** :
```python
# Sauvegarde automatique dans benchmark_runner.py
save_benchmark(results)

# Chargement pour analyse ultérieure
results = load_benchmark()
```

---

### 7. `visualize_benchmark.py` - Visualisations

**Rôle** : Génère les graphiques d'analyse des performances

**Fonctions disponibles** :

#### Comparaison de tailles (seuil AUTO)
```python
plot_size_comparison(df, save_path="graphs/size_comparison.png")
```
- Distance totale vs taille
- Durée totale vs taille  
- Runtime vs taille
- Courbes GA vs NN2O

#### Stabilité par solveur
```python
plot_stability(df, solver_name="ga", save_path="graphs/stability_ga.png")
plot_order_stability(df, solver_name="nn2o", save_path="graphs/order_stability_nn2o.png")
```
- Boxplots par matrice OSRM
- Analyse des signatures d'ordre
- Distribution des performances

#### Tuning GA
```python
plot_ga_tuning(df_ga_tuning, save_path="graphs/ga_tuning.png")
```
- Runtime par configuration
- Distance par configuration
- Ratio gain/coût

---

### 8. `run_benchmark.py` - Exécution Complète

**Rôle** : Script principal qui orchestre un benchmark complet

**Étapes d'exécution** :
1. **Configuration** : Paramètres du benchmark (commune, catégories, tailles)
2. **Préparation** : Chargement pipeline, filtrage POIs, clustering
3. **Matrices OSRM** : Génération des matrices walk/bike/car + perturbation
4. **Benchmark** : Appel à `run_benchmark()` avec tous les paramètres
5. **Visualisation** : Génération automatique des graphiques

**Paramètres modifiables** :
```python
COMMUNE = "Paris"
MAIN_CATEGORIES = ["Patrimoine & Monuments", "Gastronomie & Restauration", ...]
CLUSTER_SIZES = [2, 10, 25, 50]
RUNS_PER_MATRIX = 5
RUNS_PER_SIZE = 10
```

## Guide d'Utilisation

### Lancement Rapide

```bash
# Depuis la racine du projet
cd src/
python run_benchmark.py
```

### Résultats Attendus

**Console** :
```
→ Benchmark seuil AUTO (GA vs NN2O, tailles 2 → 50)…
→ Benchmark stabilité GA (matrices × tailles)…
→ Benchmark stabilité NN2O (matrices × tailles)…
→ Sauvegarde des résultats…
✓ Sauvegardé : benchmark_results/size_comparison.parquet
✓ Sauvegardé : benchmark_results/stability_ga.parquet
✓ Sauvegardé : benchmark_results/stability_nn2o.parquet
```

**Fichiers générés** :
```
benchmark_results/
├── size_comparison.parquet     # Comparaison GA vs NN2O
├── stability_ga.parquet        # Stabilité GA
└── stability_nn2o.parquet     # Stabilité NN2O

data/graphs/
├── size_comparison.png         # Courbes seuil AUTO
├── stability_ga.png            # Boxplots GA
├── stability_nn2o.png          # Boxplots NN2O
├── order_stability_ga.png      # Ordres GA
└── order_stability_nn2o.png    # Ordres NN2O
```

### Analyse des Résultats

**1. Seuil AUTO** : Identifier la taille où GA surpasse NN2O
**2. Stabilité** : Vérifier la robustesse aux changements de transport
**3. Performance** : Comparer les temps d'exécution
**4. Qualité** : Analyser les distances/durées obtenues

## Bonnes Pratiques

### Avant le Benchmark
```bash
# Vérifier OSRM
curl http://localhost:5000/route/v1/driving/2.3,48.8;2.4,48.9?overview=false

# Vérifier les données
ls ../data/processed/merged_*.parquet
```

### Pendant le Benchmark
- Vérifier les logs OSRM si timeouts
- Adapter `RUNS_PER_SIZE` si trop long

### Après le Benchmark
```python
# Charger pour analyse
results = load_benchmark()

# Exporter vers CSV pour Excel
results["size_comparison"].write_csv("analysis.csv")
```

## Performance et Optimisation

### Temps d'exécution estimé
- **Petit benchmark** (sizes=[2,10], runs=3) : ~5 minutes
- **Benchmark complet** (sizes=[2,10,25,50], runs=10) : ~30-45 minutes
- **GA tuning** : ~15 minutes supplémentaires

### Optimisations possibles
1. **Parallélisation** : Les runs sont indépendants
2. **Cache matrices** : Éviter de recalculer OSRM
3. **Sampling** : Réduire `runs_per_size` pour tests rapides
4. **Tailles** : Adapter `cluster_sizes` à vos données

## Dépannage

### Problèmes courants
- **OSRM timeout** : Vérifier `http://localhost:5000`
- **Memory error** : Réduire `cluster_sizes` ou `runs_per_size`
- **Empty results** : Vérifier le chemin des données POI
- **GA fails** : Augmenter `pop_size` ou réduire `ngen`

### Logs utiles
```python
# Mode debug
import logging
logging.basicConfig(level=logging.DEBUG)

# Vérifier les résultats
print(results["size_comparison"].describe())
```

---

[Retour sur la documentation de l'API](../../app/README.md)