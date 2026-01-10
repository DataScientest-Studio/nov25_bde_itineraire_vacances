from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np


class TSPSolverBase(ABC):
    """
    Classe de base pour tous les solveurs TSP Path (départ fixe, pas de retour).
    """

    def __init__(self, distance_matrix: np.ndarray, start: int = 0, name: str = "BaseSolver"):
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        self.start = start
        self.name = name
        self.validate()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    def validate(self):
        if self.D.shape[0] != self.D.shape[1]:
            raise ValueError("La matrice doit être carrée.")
        if not (0 <= self.start < self.n):
            raise ValueError("Index de départ invalide.")
        if np.any(np.isnan(self.D)):
            raise ValueError("La matrice contient des NaN.")

    # ---------------------------------------------------------
    # Coût d'une route
    # ---------------------------------------------------------
    def route_cost(self, route: List[int]) -> float:
        return sum(self.D[route[i], route[i+1]] for i in range(len(route)-1))

    # ---------------------------------------------------------
    # Interface solveur
    # ---------------------------------------------------------
    @abstractmethod
    def solve(self) -> Tuple[List[int], float]:
        """
        Chaque solveur doit implémenter cette méthode.
        """
        pass