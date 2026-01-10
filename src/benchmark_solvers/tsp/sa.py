from .base import TSPSolverBase
from typing import List, Tuple
import numpy as np

class SA_Solver(TSPSolverBase):

    def solve(self):
        # Exemple minimal : tu peux l’enrichir
        import random
        route = list(range(self.n))
        random.shuffle(route)
        if route[0] != self.start:
            route.remove(self.start)
            route.insert(0, self.start)

        best_route = route
        best_cost = self.route_cost(route)

        T = 100.0
        alpha = 0.995

        while T > 1e-3:
            i, j = sorted(random.sample(range(1, self.n), 2))
            new_route = route[:i] + route[i:j][::-1] + route[j:]
            new_cost = self.route_cost(new_route)

            if new_cost < best_cost or random.random() < np.exp((best_cost - new_cost) / T):
                route = new_route
                best_cost = new_cost
                best_route = new_route

            T *= alpha

        return best_route, best_cost