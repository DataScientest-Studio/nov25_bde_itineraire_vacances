import numpy as np
from typing import List, Tuple


class TSPSolverBase:
    def __init__(self, distance_matrix: np.ndarray, start: int = 0, name: str = "BaseTSP"):
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        self.start = start
        self.name = name

    def route_cost(self, route: List[int]) -> float:
        return float(sum(self.D[route[i], route[i+1]] for i in range(len(route) - 1)))

    def solve(self) -> Tuple[List[int], float]:
        raise NotImplementedError