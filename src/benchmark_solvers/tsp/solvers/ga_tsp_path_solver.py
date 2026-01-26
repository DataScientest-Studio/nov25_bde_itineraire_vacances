import random
from typing import List, Tuple
from deap import base, creator, tools
from .base_tsp_path_solver import TSPSolverBase


class GA_TspPathSolver(TSPSolverBase):
    """
    GA pour TSP Path (pas de retour au départ),
    comparable à NN2Opt.
    """

    def __init__(
        self,
        distance_matrix,
        start: int = 0,
        pop_size: int = 80,
        ngen: int = 200,
        cxpb: float = 0.8,
        mutpb: float = 0.2,
    ):
        super().__init__(distance_matrix, start, name="GA_TSP_Path")

        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", random.sample, range(self.n), self.n)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate(self, individual: List[int]):
        total = 0
        for i in range(len(individual) - 1):
            total += self.D[individual[i], individual[i+1]]
        return (total,)

    def solve(self) -> Tuple[List[int], float]:
        pop = self.toolbox.population(n=self.pop_size)

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for _ in range(self.ngen):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for ind in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = offspring

        best = tools.selBest(pop, 1)[0]
        route = list(best)
        cost = best.fitness.values[0]
        return route, cost