import random
import numpy as np


class NN2OptAlgo:
    """
    Implémentation simple et robuste du TSP :
    - Nearest Neighbor pour construire une solution initiale
    - 2-opt pour améliorer la solution
    """

    def __init__(self, poi_df, duration_matrix):
        """
        poi_df : DataFrame pandas contenant les POIs du cluster
        duration_matrix : matrice locale NxN (durées ou distances)
        """
        self.poi_df = poi_df.reset_index(drop=True)
        self.matrix = np.array(duration_matrix)
        self.n = len(self.poi_df)

    # ---------------------------------------------------------
    # Nearest Neighbor : construit une solution initiale
    # ---------------------------------------------------------
    def nearest_neighbor(self, start=0):
        n = self.n
        visited = [False] * n
        path = [start]
        visited[start] = True

        for _ in range(n - 1):
            last = path[-1]
            next_node = None
            best_cost = float("inf")

            for j in range(n):
                if not visited[j] and self.matrix[last][j] < best_cost:
                    best_cost = self.matrix[last][j]
                    next_node = j

            path.append(next_node)
            visited[next_node] = True

        return path

    # ---------------------------------------------------------
    # Calcul du coût d'un chemin
    # ---------------------------------------------------------
    def compute_cost(self, path):
        cost = 0
        for i in range(1, len(path)):
            cost += self.matrix[path[i - 1]][path[i]]
        return cost

    # ---------------------------------------------------------
    # 2-opt : amélioration locale
    # ---------------------------------------------------------
    def two_opt(self, path):
        improved = True
        best_path = path
        best_cost = self.compute_cost(path)

        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1:
                        continue

                    new_path = best_path[:]
                    new_path[i:j] = reversed(new_path[i:j])
                    new_cost = self.compute_cost(new_path)

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_path = new_path
                        improved = True

            path = best_path

        return best_path, best_cost

    # ---------------------------------------------------------
    # Run complet NN + 2-opt
    # ---------------------------------------------------------
    def run_nn2opt(self, try_all_starts=False):
        if self.n <= 1:
            return [], 0

        if try_all_starts:
            best_path = None
            best_cost = float("inf")

            for start in range(self.n):
                path = self.nearest_neighbor(start=start)
                path, cost = self.two_opt(path)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            return best_path, best_cost

        else:
            # Start = 0 par défaut
            path = self.nearest_neighbor(start=0)
            return self.two_opt(path)