import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tsp.base import TSPSolverBase


@dataclass
class RunResult:
    matrix: str
    solver: str
    run: int
    cost: float
    distance_km: float
    time_sec: float
    route: List[int]


class BenchmarkRunner:
    def __init__(self, start: int = 0):
        self.start = start
        self.results: List[RunResult] = []

    def run_on_matrix(self, matrix_name: str, D: np.ndarray, solver_classes, repeat=1):
        for solver_cls in solver_classes:
            solver_name = solver_cls(D, start=self.start).name

            for r in range(1, repeat + 1):
                solver = solver_cls(D, start=self.start)

                t0 = time.perf_counter()
                route, cost = solver.solve()
                t1 = time.perf_counter()

                self.results.append(
                    RunResult(
                        matrix=matrix_name,
                        solver=solver_name,
                        run=r,
                        cost=cost,
                        distance_km= cost / 1000,
                        time_sec=t1 - t0,
                        route=route,

                    )
                )

    def run_on_multiple_matrices(self, matrices: Dict[str, np.ndarray], solver_classes, repeat=1):
        for name, D in matrices.items():
            self.run_on_matrix(name, D, solver_classes, repeat)
        return self.results

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])