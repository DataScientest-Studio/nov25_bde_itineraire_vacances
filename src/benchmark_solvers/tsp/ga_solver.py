import numpy as np
import random
from typing import List, Tuple, Dict, Any
from .base import TSPSolverBase


class GASolver(TSPSolverBase):
    """
    Solveur TSP Path basé sur un algorithme génétique simple.
    - Départ fixe (start)
    - Pas de retour
    - Fonction de coût = matrice OSRM

    Hyperparamètres :
    - population_size
    - generations
    - mutation_rate
    - elite_ratio
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        start: int = 0,
        population_size: int = 80,
        generations: int = 200,
        mutation_rate: float = 0.10,
        elite_ratio: float = 0.10,
        name: str = "GA",
    ):
        super().__init__(distance_matrix, start, name=name)

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio

        # indices des POIs sauf le start
        self.nodes = [i for i in range(self.n) if i != start]

    # ---------------------------------------------------------
    # Génération initiale
    # ---------------------------------------------------------
    def _init_population(self) -> List[List[int]]:
        population = []
        for _ in range(self.population_size):
            perm = self.nodes.copy()
            random.shuffle(perm)
            route = [self.start] + perm
            population.append(route)
        return population

    # ---------------------------------------------------------
    # Sélection (roulette wheel)
    # ---------------------------------------------------------
    def _select(self, population: List[List[int]], fitness: List[float]) -> List[int]:
        total_fit = sum(fitness)
        pick = random.uniform(0, total_fit)
        current = 0
        for route, fit in zip(population, fitness):
            current += fit
            if current >= pick:
                return route
        return population[-1]

    # ---------------------------------------------------------
    # Crossover (ordre préservé)
    # ---------------------------------------------------------
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        # On ne touche pas au start
        p1 = parent1[1:]
        p2 = parent2[1:]

        a, b = sorted(random.sample(range(len(p1)), 2))
        child_middle = p1[a:b]

        child_rest = [x for x in p2 if x not in child_middle]

        child = [self.start] + child_rest[:a] + child_middle + child_rest[a:]
        return child

    # ---------------------------------------------------------
    # Mutation (swap)
    # ---------------------------------------------------------
    def _mutate(self, route: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1, self.n), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # ---------------------------------------------------------
    # Fitness = 1 / cost
    # ---------------------------------------------------------
    def _compute_fitness(self, population: List[List[int]]) -> List[float]:
        fitness = []
        for route in population:
            cost = self.route_cost(route)
            fitness.append(1.0 / (cost + 1e-9))
        return fitness

    # ---------------------------------------------------------
    # Boucle GA
    # ---------------------------------------------------------
    def _run_ga(self) -> List[int]:
        population = self._init_population()

        elite_count = max(1, int(self.elite_ratio * self.population_size))

        for _ in range(self.generations):
            fitness = self._compute_fitness(population)

            # Tri par fitness décroissante
            ranked = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            elites = [r[0] for r in ranked[:elite_count]]

            # Nouvelle population
            new_pop = elites.copy()

            while len(new_pop) < self.population_size:
                parent1 = self._select(population, fitness)
                parent2 = self._select(population, fitness)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop

        # Meilleur individu final
        final_fitness = self._compute_fitness(population)
        best_idx = np.argmax(final_fitness)
        return population[best_idx]

    # ---------------------------------------------------------
    # solve()
    # ---------------------------------------------------------
    def solve(self) -> Tuple[List[int], float]:
        route = self._run_ga()
        cost = self.route_cost(route)
        return route, cost