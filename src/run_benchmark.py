import polars as pl

from pipeline.itinerary_pipeline import ItineraryPipeline
from features.osrm import OSRMClientAsync  

from benchmark.benchmark_runner import run_benchmark
from benchmark.visualize_benchmark import (
    plot_size_comparison,
    plot_stability,
    plot_order_stability,
    plot_ga_tuning

)


# ---------------------------------------------------------
# 1. PARAMÈTRES DU BENCHMARK
# ---------------------------------------------------------

COMMUNE = "Paris"
MAIN_CATEGORIES = ["Patrimoine & Monuments", 
                        "Gastronomie & Restauration",
                        "Culture & Musées",
                        "Commerce & Shopping",
                        "Camping & Plein Air",
                        "Famille & Enfants",
                        "Nature & Paysages",
                        "Sports & Loisirs",
                        "Bien-être & Santé",
                        "Loisirs & Clubs"]

SUB_CATEGORIES = ["Restaurants","Bibliothèques & médiation","Restauration rapide","Bars & cafés",
                    "Religieux","Sports collectifs & stades","Parcs & loisirs","Cimet,ières & mémoriaux",
                    "Zoo & animaux","Théâtres & cinémas","Salles de concert & clubs","Commerces"
                    ]
MIN_SCORE = 0.15
NB_DAYS = 5
ANCHOR_LAT = 48.86666
ANCHOR_LON = 2.33333

TRANSPORT_MODE = "walk"  # pour préparer les POIs
INPUT_PATH = "../data/processed/merged_20260108_174125.parquet"

# tailles de clusters pour robustesse
CLUSTER_SIZES = [2, 10, 25, 50]

# répétitions par matrice
RUNS_PER_MATRIX = 3

# répétitions par taille pour seuil AUTO
RUNS_PER_SIZE = 3


# ---------------------------------------------------------
# 2. PIPELINE + OSRM
# ---------------------------------------------------------

print("Chargement du pipeline…")
osrm = OSRMClientAsync("http://localhost:5000")
pipeline = ItineraryPipeline(pois_path=INPUT_PATH)

print("Filtrage des POIs…")
filtered_lf = pipeline._filter_pois(
    COMMUNE, MAIN_CATEGORIES, SUB_CATEGORIES, MIN_SCORE
)

print("Clustering…")
df_prepared = pipeline._cluster_pois(
    filtered_lf, NB_DAYS, ANCHOR_LAT, ANCHOR_LON
).collect()

print("Préparation OSRM-ready…")
df_clustered = pipeline._build_osrm_ready_pois(
    df_prepared=df_prepared,
    mode=TRANSPORT_MODE,
)


# ---------------------------------------------------------
# 3. PRÉPARER LES POIs
# ---------------------------------------------------------

print("Génération matrices OSRM WALK…")
df_clustered, dist_walk, dur_walk = pipeline._compute_osrm_matrices(
    df_clustered=df_clustered,
    osrm=osrm,
    transport_mode="walk",
)

print("Génération matrices OSRM BIKE…")
df_clustered, dist_bike, dur_bike = pipeline._compute_osrm_matrices(
    df_clustered=df_clustered,
    osrm=osrm,
    transport_mode="bike",
)

print("Génération matrices OSRM CAR…")
df_clustered, dist_car, dur_car = pipeline._compute_osrm_matrices(
    df_clustered=df_clustered,
    osrm=osrm,
    transport_mode="car",
)



print("Création matrice WALK perturbée…")
dist_walk_pert = dist_walk * 1.02
dur_walk_pert = dur_walk * 1.02

matrices = {
    "walk": {"dur": dur_walk, "dist": dist_walk},
    "bike": {"dur": dur_bike, "dist": dist_bike},
    "car": {"dur": dur_car, "dist": dist_car},
    "walk_perturbed": {"dur": dur_walk_pert, "dist": dist_walk_pert},
}


# ---------------------------------------------------------
# 4. SOLVEURS
# ---------------------------------------------------------

compute_ga = pipeline._compute_itinerary_ga
compute_nn2o = pipeline._compute_itinerary_nn2o


# ---------------------------------------------------------
# 5. BENCHMARK
# ---------------------------------------------------------

print("\n>>> Appel à run_benchmark()…\n")

results = run_benchmark(
    df_all_pois=df_clustered,
    matrices=matrices,
    compute_ga=compute_ga,
    compute_nn2o=compute_nn2o,
    pipeline=pipeline,
    cluster_sizes=CLUSTER_SIZES,
    runs_per_matrix=RUNS_PER_MATRIX,
    runs_per_size=RUNS_PER_SIZE,
)


df_size_comp = results["size_comparison"]
df_stab_ga = results["stability_ga"]
df_stab_nn2o = results["stability_nn2o"]
df_ga_tuning = results["ga_tuning"]


print("\n=== COMPARAISON PAR TAILLES (SEUIL AUTO) ===")
print(df_size_comp)

print("\n=== STABILITÉ GA ===")
print(df_stab_ga)

print("\n=== STABILITÉ NN2O ===")
print(df_stab_nn2o)

print("\n=== TUNING GA ===")
print(df_ga_tuning)


print("\n>>> Résultats reçus !\n")

# ---------------------------------------------------------
# 6. VISUALISATIONS
# ---------------------------------------------------------

print("\n>>> Appel aux visualisations…\n")

#results = load_benchmark()

print("\n>>> Graphiques seuil AUTO")
plot_size_comparison(
    results["size_comparison"],
    save_path="../data/graphs/size_comparison.png"
)

plot_stability(
    results["stability_ga"],
    "ga",
    save_path="../data/graphs/stability_ga.png"
)

plot_order_stability(
    results["stability_ga"],
    "ga",
    save_path="../data/graphs/order_stability_ga.png"
)

plot_stability(
    results["stability_nn2o"],
    "nn2o",
    save_path="../data/graphs/stability_nn2o.png"
)

plot_order_stability(
    results["stability_nn2o"],
    "nn2o",
    save_path="../data/graphs/order_stability_nn2o.png"
)


plot_ga_tuning(
    results["ga_tuning"],
    save_path="../data/graphs/ga_tuning.png"
)
print("\n>>> Visualisations terminées !\n")
