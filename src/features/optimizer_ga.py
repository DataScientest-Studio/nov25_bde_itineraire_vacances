from deap import base
from deap import creator
from deap import tools

import random
import numpy as np
import pandas as pd


class GeneticAlgo:
    """
    Algorithme génétique pour optimiser un itinéraire de POIs.
    Travaille en indices (0..N-1) pour être compatible avec OSRM.
    Retourne ensuite les poi_id réels.
    """

    # -------------------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------------------
    def __init__(self, poi_df: pd.DataFrame, duration_matrix: np.ndarray):

        # Nettoyage des classes DEAP existantes
        if hasattr(creator, "FitnessItinerary"):
            del creator.FitnessItinerary
        if hasattr(creator, "Itinerary"):
            del creator.Itinerary

        creator.create("FitnessItinerary", base.Fitness, weights=(1.0,))
        creator.create("Itinerary", list, fitness=creator.FitnessItinerary)

        self.toolbox = base.Toolbox()

        # Copie du DF
        self.df = poi_df.copy()

        # Garantir la présence de sub_category
        if "sub_category" not in self.df.columns:
            self.df["sub_category"] = "Unknown"
        else:
            self.df["sub_category"] = self.df["sub_category"].fillna("Unknown")

        # Matrice OSRM NxN
        self.matrix = duration_matrix

        # Mapping indices ↔ poi_id
        self.poi_ids = self.df["poi_id"].tolist()
        self.index_to_poi = {i: pid for i, pid in enumerate(self.poi_ids)}
        self.poi_to_index = {pid: i for i, pid in enumerate(self.poi_ids)}

    # -------------------------------------------------------------------------
    # REPAIR FUNCTION (ANTI-DOUBLONS)
    # -------------------------------------------------------------------------
    def repair_itinerary(self, itin):
        """
        Supprime les doublons et remplace les trous par des indices non utilisés.
        Garantit un individu valide et unique.
        """
        n = len(self.poi_ids)
        used = set()
        new_itin = []

        # Garder les éléments uniques dans l'ordre
        for idx in itin:
            if idx not in used:
                new_itin.append(idx)
                used.add(idx)

        # Compléter avec des indices manquants
        missing = [i for i in range(n) if i not in used]
        random.shuffle(missing)

        while len(new_itin) < len(itin):
            new_itin.append(missing.pop())

        return creator.Itinerary(new_itin)

    # -------------------------------------------------------------------------
    # DURÉES & SCORES
    # -------------------------------------------------------------------------
    def get_itinerary_travel_duration(self, itin_indices):
        if len(itin_indices) < 2:
            return 0.0
        return sum(
            self.matrix[itin_indices[i], itin_indices[i + 1]]
            for i in range(len(itin_indices) - 1)
        )

    def get_itinerary_resto(self, itin_indices, resto_cat=None):
        if resto_cat is None:
            resto_cat = ["Restaurants"]

        poi_ids = [self.index_to_poi[i] for i in itin_indices]
        df_itin = self.df[self.df["poi_id"].isin(poi_ids)]
        df_resto = df_itin[df_itin["sub_category"].isin(resto_cat)]
        return df_resto["poi_id"].tolist()

    def get_itinerary_activity_duration(
        self,
        itin_indices,
        lunch_duration=60,
        activity_duration=45,
        resto_cat=None,
    ):
        if resto_cat is None:
            resto_cat = ["Restaurants"]

        itin_resto = self.get_itinerary_resto(itin_indices, resto_cat)
        poi_ids = [self.index_to_poi[i] for i in itin_indices]

        return sum(
            lunch_duration if pid in itin_resto else activity_duration
            for pid in poi_ids
        )

    def get_itinerary_duration_score(self, itin_indices, duration=8):
        total_minutes = (
            self.get_itinerary_activity_duration(itin_indices)
            + self.get_itinerary_travel_duration(itin_indices)
        )
        hours = total_minutes / 60.0
        return float(np.exp(-(hours - duration) ** 2))

    def get_lunch_time(self, itin_indices, start_time=9, resto_cat=None):
        if resto_cat is None:
            resto_cat = ["Restaurants"]

        itin_resto = self.get_itinerary_resto(itin_indices, resto_cat)
        if len(itin_resto) == 0:
            return 0.0

        poi_ids = [self.index_to_poi[i] for i in itin_indices]
        first_resto = itin_resto[0]
        idx = poi_ids.index(first_resto)

        travel_before = self.get_itinerary_travel_duration(
            itin_indices[: idx + 1]
        ) / 60.0
        activity_before = (
            self.get_itinerary_activity_duration(itin_indices[:idx]) / 60.0
        )

        return start_time + travel_before + activity_before

    def get_itinerary_resto_score(
        self,
        itin_indices,
        resto_cat=None,
        start_time=9,
        lunch_time=13,
    ):
        if resto_cat is None:
            resto_cat = ["Restaurants"]

        itin_resto = self.get_itinerary_resto(itin_indices, resto_cat)
        resto_nbre = len(itin_resto)
        L = len(itin_indices)

        if L == 0:
            return 0.0

        if resto_nbre == 0:
            resto_score = np.exp(L - 2) / np.exp(L)
        else:
            resto_score = np.exp(L - resto_nbre) / np.exp(L)

        lunch_t = self.get_lunch_time(itin_indices, start_time, resto_cat)
        lunch_score = float(np.exp(-(lunch_t - lunch_time) ** 2))

        return 0.7 * lunch_score + 0.3 * resto_score

    def evaluate_itinerary(
        self,
        itin_indices,
        duration=8,
        resto_cat=None,
        start_time=9,
        lunch_time=13,
    ):
        if resto_cat is None:
            resto_cat = ["Restaurants"]

        resto_score = self.get_itinerary_resto_score(
            itin_indices, resto_cat, start_time, lunch_time
        )
        duration_score = self.get_itinerary_duration_score(
            itin_indices, duration
        )

        return (0.6 * duration_score + 0.4 * resto_score,)

    # -------------------------------------------------------------------------
    # CROSSOVER & MUTATION (AVEC RÉPARATION)
    # -------------------------------------------------------------------------
    def crossover_itinerary(self, itin1, itin2):
        size = min(len(itin1), len(itin2))
        if size < 2:
            return itin1, itin2

        p1, p2 = sorted(random.sample(range(size), 2))

        cr1 = itin1[p1:p2]
        cr2 = itin2[p1:p2]

        itin1[p1:p2] = cr2
        itin2[p1:p2] = cr1

        return self.repair_itinerary(itin1), self.repair_itinerary(itin2)

    def mutate_itinerary(self, itin):
        if len(itin) < 2:
            return (itin,)

        p1, p2 = random.sample(range(len(itin)), 2)
        itin[p1], itin[p2] = itin[p2], itin[p1]

        return (self.repair_itinerary(itin),)

    # -------------------------------------------------------------------------
    # TOOLBOX
    # -------------------------------------------------------------------------
    def generate_random_itinerary(self, itin_min_poi=5, itin_max_poi=15):
        n = len(self.poi_ids)
        size = random.randint(itin_min_poi, min(itin_max_poi, n))
        itin = random.sample(range(n), k=size)
        return self.repair_itinerary(itin)

    def setup_toolbox(self, itin_min_poi=5, itin_max_poi=15):
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "itinerary",
            self.generate_random_itinerary,
            itin_min_poi,
            itin_max_poi,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.itinerary)
        self.toolbox.register("evaluate", self.evaluate_itinerary)
        self.toolbox.register("mate", self.crossover_itinerary)
        self.toolbox.register("mutate", self.mutate_itinerary)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # -------------------------------------------------------------------------
    # BOUCLE PRINCIPALE AVEC ÉLITISME
    # -------------------------------------------------------------------------
    def run_ga(self, pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3):
        pop = self.toolbox.population(n=pop_size)

        # Évaluation initiale
        for itin in pop:
            itin.fitness.values = self.toolbox.evaluate(itin)

        for gen in range(ngen):

            # 🔥 ÉLITISME : on garde le meilleur
            elite = max(pop, key=lambda x: x.fitness.values[0])
            elite_clone = self.toolbox.clone(elite)

            # Sélection
            offspring = self.toolbox.select(pop, len(pop) - 1)
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Crossover
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values

            # Mutation
            for ind in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            # Réévaluation
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Nouvelle population = élite + offspring
            pop = [elite_clone] + offspring

        # Meilleur individu final
        best = max(pop, key=lambda x: x.fitness.values[0])
        best_indices = list(best)
        best_poi_ids = [self.index_to_poi[i] for i in best_indices]

        return best_poi_ids, best.fitness.values[0]