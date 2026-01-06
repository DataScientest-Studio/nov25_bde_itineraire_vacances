from typing import List
import math


class TSPSolver:
    """
    Résout un TSP simple :
    - point de départ fixe (index 0 = anchor)
    - Nearest Neighbor pour solution initiale
    - 2-opt pour amélioration
    """

    def __init__(self, matrix: List[List[float]]):
        """
        matrix : matrice NxN des durées (ou distances)
        """
        self.matrix = matrix
        self.n = len(matrix)

    # ---------------------------------------------------------
    # Nearest Neighbor (solution initiale)
    # ---------------------------------------------------------

    def nearest_neighbor(self) -> List[int]:
        """
        Retourne un ordre initial en partant du point 0 (anchor).
        """
        visited = [False] * self.n
        path = [0]
        visited[0] = True

        for _ in range(self.n - 1):
            last = path[-1]
            best = None
            best_cost = math.inf

            for j in range(self.n):
                if not visited[j] and self.matrix[last][j] < best_cost:
                    best = j
                    best_cost = self.matrix[last][j]

            path.append(best)
            visited[best] = True

        return path

    # ---------------------------------------------------------
    # 2-opt (amélioration)
    # ---------------------------------------------------------

    def two_opt(self, path: List[int]) -> List[int]:
        """
        Améliore un chemin avec l'algorithme 2-opt.
        """
        improved = True

        while improved:
            improved = False

            for i in range(1, self.n - 2):
                for j in range(i + 1, self.n - 1):

                    # Coût actuel
                    a, b = path[i - 1], path[i]
                    c, d = path[j], path[j + 1]

                    current = self.matrix[a][b] + self.matrix[c][d]
                    swapped = self.matrix[a][c] + self.matrix[b][d]

                    if swapped < current:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True

        return path

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    def solve(self) -> List[int]:
        """
        Retourne l'ordre optimal approximé :
        - commence par NN
        - améliore avec 2-opt
        """
        initial = self.nearest_neighbor()
        improved = self.two_opt(initial)
        return improved