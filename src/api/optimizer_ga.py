import random

import numpy as np
from deap import base, creator, tools


class GeneticAlgo:
    def __init__(self, poi_df, duration_matrix):
        """
        GA pour itinéraires touristiques.
        - poi_df : DataFrame avec au moins osrm_index, sub_category
        - duration_matrix : matrice OSRM (numpy array ou DataFrame)
        """

        # Reset DEAP classes si déjà créées
        if hasattr(creator, "FitnessItinerary"):
            del creator.FitnessItinerary
        if hasattr(creator, "Itinerary"):
            del creator.Itinerary

        creator.create("FitnessItinerary", base.Fitness, weights=(1.0,))
        creator.create("Itinerary", list, fitness=creator.FitnessItinerary)

        self.toolbox = base.Toolbox()
        self.df = poi_df
        self.matrix = duration_matrix  # numpy array ou DataFrame

        # On travaille UNIQUEMENT avec osrm_index pour être aligné avec la matrice
        self.poi_list = self.df["osrm_index"].tolist()

    # --------------------------------------------------------------------------
    # UTILITAIRE : accès robuste à la matrice
    # --------------------------------------------------------------------------
    def _get(self, i, j):
        """Accès compatible numpy ou pandas."""
        if hasattr(self.matrix, "at"):
            return self.matrix.at[i, j]
        return self.matrix[i, j]

    # --------------------------------------------------------------------------
    # DURÉES & RESTO
    # --------------------------------------------------------------------------
    def get_itinerary_travel_duration(self, itin):
        """
        Durée totale de déplacement (en minutes) pour un itinéraire d'osrm_index.
        """
        if len(itin) < 2:
            return 0.0
        return sum(self._get(itin[i], itin[i + 1]) for i in range(len(itin) - 1))

    def get_itinerary_resto(self, itin, resto_cat=["Restaurants"]):
        """
        Retourne les osrm_index des POIs de l'itinéraire qui sont des restos.
        """
        df_itin = self.df.loc[self.df.osrm_index.isin(itin)]
        itin_resto = df_itin.loc[df_itin.sub_category.isin(resto_cat), "osrm_index"]
        return itin_resto.tolist()

    def get_itinerary_activity_duration(
        self,
        itin,
        lunch_duration=60,
        activity_duration=45,
        resto_cat=["Restaurants"],
    ):
        """
        Durée totale des activités (en minutes) pour un itinéraire.
        - resto : lunch_duration
        - autres : activity_duration
        """
        itin_resto = self.get_itinerary_resto(itin, resto_cat=resto_cat)
        return sum(
            lunch_duration if poi in itin_resto else activity_duration for poi in itin
        )

    def get_itinerary_duration_score(self, itin, duration=8):
        """
        Score basé sur la durée totale de la journée (heures).
        """
        total_minutes = self.get_itinerary_activity_duration(
            itin
        ) + self.get_itinerary_travel_duration(itin)
        itin_duration_hours = total_minutes / 60.0
        return float(np.exp(-((itin_duration_hours - duration) ** 2)))

    def get_lunch_time(self, itin, start_time=9):
        """
        Heure du déjeuner (en heures) si resto dans l'itinéraire, sinon 0.
        """
        itin_resto = self.get_itinerary_resto(itin, resto_cat=["Restaurants"])
        if len(itin_resto) == 0:
            return 0.0

        # On prend le premier resto dans l'itinéraire
        first_resto = itin_resto[0]
        if first_resto not in itin:
            return 0.0

        resto_idx = itin.index(first_resto)

        travel_before = self.get_itinerary_travel_duration(itin[: resto_idx + 1]) / 60.0
        activity_before = self.get_itinerary_activity_duration(itin[:resto_idx]) / 60.0

        return float(start_time + travel_before + activity_before)

    def get_itinerary_resto_score(
        self,
        itin,
        resto_cat=["Restaurants"],
        start_time=9,
        lunch_time=13,
    ):
        """
        Score combinant :
        - nombre de restos
        - heure du déjeuner
        """
        itin_resto = self.get_itinerary_resto(itin, resto_cat=resto_cat)
        resto_nbre = len(itin_resto)

        # Score sur le nombre de restos
        if resto_nbre == 0:
            resto_score = np.exp(len(itin) - 2) / np.exp(len(itin))
        else:
            resto_score = np.exp(len(itin) - resto_nbre) / np.exp(len(itin))

        # Score sur l'heure du déjeuner
        itin_lunch_time = self.get_lunch_time(itin, start_time=start_time)
        lunch_score = np.exp(-((itin_lunch_time - lunch_time) ** 2))

        return float(0.7 * lunch_score + 0.3 * resto_score)

    # --------------------------------------------------------------------------
    # FITNESS
    # --------------------------------------------------------------------------
    def evaluate_itinerary(
        self,
        itin,
        duration=8,
        resto_cat=["Restaurants"],
        start_time=9,
        lunch_time=13,
    ):
        """
        Fitness globale de l'itinéraire.
        """
        if len(itin) == 0:
            return (0.0,)

        resto_score = self.get_itinerary_resto_score(
            itin,
            resto_cat=resto_cat,
            start_time=start_time,
            lunch_time=lunch_time,
        )
        duration_score = self.get_itinerary_duration_score(itin, duration=duration)

        return (0.6 * duration_score + 0.4 * resto_score,)

    # --------------------------------------------------------------------------
    # CROSSOVER & MUTATION
    # --------------------------------------------------------------------------
    def crossover_itinerary(self, itin1, itin2):
        """
        Crossover simple sur la partie commune des deux itinéraires.
        """
        size = min(len(itin1), len(itin2))
        if size < 2:
            return itin1, itin2

        p1, p2 = sorted(random.sample(range(size), 2))
        cr1 = itin1[p1:p2]
        cr2 = itin2[p1:p2]
        itin1[p1:p2] = cr2
        itin2[p1:p2] = cr1
        return itin1, itin2

    def mutate_itinerary(self, itin):
        """
        Mutation : swap de deux positions.
        """
        if len(itin) < 2:
            return (itin,)
        p1, p2 = random.sample(range(len(itin)), 2)
        itin[p1], itin[p2] = itin[p2], itin[p1]
        return (itin,)

    # --------------------------------------------------------------------------
    # INITIALISATION
    # --------------------------------------------------------------------------
    def generate_random_itinerary(self, itin_min_poi=5, itin_max_poi=150):
        """
        Génère un itinéraire aléatoire d'osrm_index.
        """
        if len(self.poi_list) <= itin_min_poi:
            itin_min_poi = len(self.poi_list)
            itin_max_poi = len(self.poi_list)

        if itin_min_poi < len(self.poi_list) < itin_max_poi:
            itin_max_poi = len(self.poi_list)

        size = random.randint(itin_min_poi, itin_max_poi)
        itin = random.sample(self.poi_list, k=size)
        return creator.Itinerary(itin)

    def setup_toolbox(self, itin_min_poi=5, itin_max_poi=150):
        """
        Configuration de la toolbox DEAP.
        """
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "itinerary",
            self.generate_random_itinerary,
            itin_min_poi,
            itin_max_poi,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.itinerary
        )
        self.toolbox.register("evaluate", self.evaluate_itinerary)
        self.toolbox.register("mate", self.crossover_itinerary)
        self.toolbox.register("mutate", self.mutate_itinerary)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # --------------------------------------------------------------------------
    # ALGORITHME GÉNÉTIQUE
    # --------------------------------------------------------------------------
    def run_ga(self, pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3):
        """
        Boucle principale de l'algo génétique.
        Retourne :
        - best_itinerary : liste d'osrm_index
        - best_fitness : score de l'itinéraire
        """
        pop = self.toolbox.population(n=pop_size)

        # Évaluation initiale
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for itin, fit in zip(pop, fitnesses):
            itin.fitness.values = fit

        for gen in range(ngen):
            # Sélection
            offspring = self.toolbox.select(pop, len(pop))
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
            fitnesses = map(self.toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = offspring

        best = max(pop, key=lambda x: x.fitness.values[0])
        return list(best), float(best.fitness.values[0])
