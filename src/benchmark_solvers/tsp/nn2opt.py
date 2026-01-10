import numpy as np
from typing import List, Tuple
from .base import TSPSolverBase


class NN2OptSolver(TSPSolverBase):
    """
    Solveur TSP Path : Nearest Neighbor + 2-opt
    """

    def __init__(self, distance_matrix: np.ndarray, start: int = 0):
        super().__init__(distance_matrix, start, name="NN2Opt")

    # ---------------------------------------------------------
    # Construction initiale : Nearest Neighbor
    # ---------------------------------------------------------
    def nearest_neighbor(self) -> List[int]:
        unvisited = set(range(self.n))
        unvisited.remove(self.start)

        route = [self.start]
        current = self.start

        while unvisited:
            next_node = min(unvisited, key=lambda j: self.D[current, j])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return route

    # ---------------------------------------------------------
    # 2-opt adapté au TSP path (pas de retour)
    # ---------------------------------------------------------
    def two_opt(self, route: List[int]) -> List[int]:
        improved = True
        best_route = route
        best_cost = self.route_cost(route)

        while improved:
            improved = False
            for i in range(1, self.n - 2):
                for k in range(i + 1, self.n - 1):
                    new_route = (
                        best_route[:i]
                        + best_route[i:k+1][::-1]
                        + best_route[k+1:]
                    )
                    new_cost = self.route_cost(new_route)

                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return best_route

    # ---------------------------------------------------------
    # Pipeline complet
    # ---------------------------------------------------------
    def solve(self) -> Tuple[List[int], float]:
        route = self.nearest_neighbor()
        route = self.two_opt(route)
        cost = self.route_cost(route)
        return route, cost