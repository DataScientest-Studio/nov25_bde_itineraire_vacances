from benchmark.benchmark_ga_tuning import run_ga_tuning
import polars as pl

from pipeline.itinerary_pipeline import ItineraryPipeline
from features.osrm import OSRMClientAsync  
from benchmark.visualize_ga_tuning import plot_ga_tuning, plot_runtime, plot_fitness, plot_tradeoff

from benchmark.benchmark_ga_tuning import run_ga_tuning
from benchmark.ga_selection import select_best_ga_config
from benchmark.benchmark_io import save_benchmark

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
RUNS_PER_MATRIX = 4

# répétitions par taille pour seuil AUTO
RUNS_PER_SIZE = 4


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

print(df_clustered.columns)

print("Ajout de la colonne cluster_size…")
df_clustered = df_clustered.with_columns(
    pl.len().over("cluster_id").alias("cluster_size")
)

print('AFTER creation col',df_clustered.columns)

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

# ---------------------------------------------------------
# 5. BENCHMARK
# ---------------------------------------------------------

transport_mode = "car"

df_ga_tuning = run_ga_tuning(
    df_clustered=df_clustered,
    matrices=matrices,
    matrix_id=transport_mode,
    runs_per_cluster=4,
)

print(df_ga_tuning)
# save results
results = {
    f"ga_tuning_{transport_mode}": df_ga_tuning,
}
save_benchmark(results)


plot_ga_tuning(df_ga_tuning, save_path=f"../data/graphs/ga_tuning_{transport_mode}.png")
plot_runtime(df_ga_tuning, save_path=f"../data/graphs/ga_runtime_{transport_mode}.png")
plot_fitness(df_ga_tuning, save_path=f"../data/graphs/ga_fitness_{transport_mode}.png")
plot_tradeoff(df_ga_tuning, save_path=f"../data/graphs/ga_tradeoff_{transport_mode}.png")


best_config, summary = select_best_ga_config(df_ga_tuning)

print("Meilleure configuration GA :", best_config)

