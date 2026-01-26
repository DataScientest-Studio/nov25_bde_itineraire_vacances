import time
import numpy as np
from typing import List, Type

from benchmark_solvers.tsp.solvers.base_tsp_path_solver import TSPSolverBase
from benchmark_solvers.tsp.solvers.nn2opt_tsp_path_solver import NN2OptSolver
from benchmark_solvers.tsp.solvers.ga_tsp_path_solver import GA_TspPathSolver

from .tsp_metrics import (
    basic_stats,
    stability_metrics,
    quality_metrics,
    structural_metrics,
    pareto_score,
    tsp_rating,
)


class TSPBenchmarkRunner:
    def __init__(self, datasets, start=0, runs=10):
        """
        datasets : dict {
            "14_pois": { "matrix": np.array, "size": 14 },
            "29_pois": { "matrix": np.array, "size": 29 },
            ...
        }
        """
        self.datasets = datasets
        self.start = start
        self.runs = runs

        self.solver_classes: List[Type[TSPSolverBase]] = [
            NN2OptSolver,
            GA_TspPathSolver,
        ]

    def run(self):
        all_results = []

        for matrix_name, dataset in self.datasets.items():
            D = dataset["matrix"]
            size = dataset["size"]

            print(f"\n--- Benchmark TSP on dataset: {matrix_name} ({size} POIs) ---")

            global_best_cost = float("inf")
            solver_runs = {}

            # ------------------------------------------------------------
            # 1) Exécuter tous les solveurs
            # ------------------------------------------------------------
            for solver_cls in self.solver_classes:
                costs = []
                times = []
                routes = []

                for _ in range(self.runs):
                    solver = solver_cls(D, start=self.start)

                    t0 = time.time()
                    route, cost = solver.solve()
                    t1 = time.time()

                    costs.append(cost)
                    times.append(t1 - t0)
                    routes.append(route)

                solver_runs[solver_cls.__name__] = {
                    "costs": costs,
                    "times": times,
                    "routes": routes,
                }

                global_best_cost = min(global_best_cost, min(costs))

            # ------------------------------------------------------------
            # 2) Calcul des métriques pour chaque solveur
            # ------------------------------------------------------------
            for solver_name, data in solver_runs.items():
                costs = data["costs"]
                times = data["times"]
                routes = data["routes"]

                base = basic_stats(costs, times)
                stab = stability_metrics(costs)
                qual = quality_metrics(costs, global_best_cost)

                best_route = routes[np.argmin(costs)]
                struct = structural_metrics(best_route, D)

                pareto = pareto_score(
                    base["mean_cost"],
                    base["mean_time"],
                    max_cost=base["max_cost"],
                    max_time=base["max_time"],
                )

                rating = tsp_rating(
                    efficiency=qual["efficiency"],
                    robustness=stab["robustness"],
                    pareto=pareto,
                )

                result = {
                    "matrix": matrix_name,
                    "size": size,
                    "solver": solver_name,
                    **base,
                    **stab,
                    **qual,
                    **struct,
                    "pareto_score": pareto,
                    "rating": rating,
                    "best_route": best_route,
                }

                all_results.append(result)

        return all_results