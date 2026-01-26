import random
import numpy as np
from deap import base, creator, tools
from .base import TSPSolverBase


class GATspPathSolver(TSPSolverBase):
    """
    Solveur GA pour le TSP Path (pas de retour au point de départ).
    Compatible avec NN2Opt et ton BenchmarkRunner.
    """

    def __init__(self, D, start=0, pop_size=80, ngen=200, cxpb=0.8, mutpb=0.2):
        super().__init__(D, start)
        self.name = "GA_TSP_Path"

        self.n = D.shape[0]
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

        # Création des classes DEAP
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox
        self.toolbox = base.Toolbox()

        # Individu = permutation complète
        self.toolbox.register("indices", random.sample, range(self.n), self.n)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Évaluation
        self.toolbox.register("evaluate", self.evaluate)

        # Crossover OX
        self.toolbox.register("mate", tools.cxOrdered)

        # Mutation : shuffle
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

        # Sélection
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # ---------------------------------------------------------
    # Fitness = distance totale SANS retour au départ
    # ---------------------------------------------------------
    def evaluate(self, individual):
        D = self.D
        total = 0
        for i in range(len(individual) - 1):
            total += D[individual[i], individual[i+1]]
        return (total,)

    # ---------------------------------------------------------
    # Solveur principal
    # ---------------------------------------------------------
    def solve(self):
        pop = self.toolbox.population(n=self.pop_size)

        # Évaluation initiale
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Boucle GA
        for gen in range(self.ngen):

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutation
            for ind in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            # Réévaluation
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = offspring

        # Meilleur individu
        best = tools.selBest(pop, 1)[0]
        route = list(best)
        cost = best.fitness.values[0]

        return route, cost