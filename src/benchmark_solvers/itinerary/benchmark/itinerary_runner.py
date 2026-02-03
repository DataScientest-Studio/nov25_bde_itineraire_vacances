import time

from benchmark_solvers.itinerary.solvers.ga_itinerary_path_solver import (
    GAItinerarySolver,
)
from benchmark_solvers.itinerary.solvers.nn2opt_itinerary_solver import (
    NN2OptItinerarySolver,
)

from .itinerary_metrics import itinerary_metrics


class ItineraryBenchmarkRunner:
    def __init__(self, datasets, runs=10):
        self.datasets = datasets
        self.runs = runs
        self.solvers = [GAItinerarySolver, NN2OptItinerarySolver]

    def run(self):
        results = []

        for name, data in self.datasets.items():
            poi_df = data["poi_df"]
            matrix = data["matrix"]
            size = data["size"]

            print(f"\n--- Itinerary benchmark on {name} ({size} POIs) ---")

            for solver_cls in self.solvers:
                scores, times, routes = [], [], []

                for _ in range(self.runs):
                    solver = solver_cls(poi_df, matrix)
                    t0 = time.time()
                    route, score = solver.solve()
                    t1 = time.time()

                    scores.append(score)
                    times.append(t1 - t0)
                    routes.append(route)

                metrics = itinerary_metrics(scores, times, routes, poi_df, matrix)
                results.append(
                    {
                        "matrix": name,
                        "size": size,
                        "solver": solver_cls.__name__,
                        **metrics,
                    }
                )

        return results
