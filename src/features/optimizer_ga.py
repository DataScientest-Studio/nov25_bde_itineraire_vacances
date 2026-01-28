from deap import base
from deap import creator
from deap import tools

import random
import numpy as np


class GeneticAlgo:

    def __init__(self, poi_df, duration_matrix):

        # Supprimer les classes si elles existent déjà
        if hasattr(creator, "FitnessItinerary"):
            del creator.FitnessItinerary
        if hasattr(creator, "Itinerary"):
            del creator.Itinerary

        # intitialisation de la classs fitness = fonction de sélection :
        creator.create("FitnessItinerary", base.Fitness, weights=(1.0,))  # maximisation

        # intiialisation de la classe individu = un itinéraire :
        creator.create("Itinerary", list, fitness=creator.FitnessItinerary)

        # création de la toolbox (conteneur de toutes les opérations):
        self.toolbox = base.Toolbox()
        self.df = poi_df              # DataFrame pandas local au cluster
        self.matrix = duration_matrix # matrice locale (np.ndarray, shape (n_local, n_local))
        self.n_local = duration_matrix.shape[0]  # nombre de POIs dans le cluster

    # --------------------------------------------------------------------------
    #           Fonctions de durée / resto / score
    # --------------------------------------------------------------------------

    def get_itinerary_travel_duration(self, itin):
        """
        itin : liste d'indices LOCAUX (0..n_local-1)
        """
        if len(itin) < 2:
            return 0.0
        travel_duration = sum(
            self.matrix[itin[i], itin[i + 1]] for i in range(len(itin) - 1)
        )
        return travel_duration

    def get_itinerary_resto(self, itin, resto_cat=['Restaurants']):
        """
        itin : liste d'indices LOCAUX
        retourne les poi_id qui sont dans une des sub_categories resto_cat
        """
        if len(itin) == 0:
            return []

        # On récupère les lignes correspondant aux indices locaux
        df_itin = self.df.iloc[itin]
        itin_resto = df_itin.loc[df_itin.sub_category.isin(resto_cat), 'poi_id']
        return itin_resto.tolist()

    def get_itinerary_activity_duration(
        self,
        itin,
        lunch_duration=60,
        activity_duration=45,
        resto_cat=['Restaurants']
    ):
        """
        itin : indices LOCAUX
        """
        if len(itin) == 0:
            return 0.0

        itin_resto = self.get_itinerary_resto(itin, resto_cat=resto_cat)

        # On a besoin des poi_id pour comparer
        df_itin = self.df.iloc[itin]
        poi_ids = df_itin['poi_id'].tolist()

        activities_duration = sum(
            lunch_duration if poi in itin_resto else activity_duration
            for poi in poi_ids
        )
        return activities_duration

    def get_itinerary_duration_score(self, itin, duration=8):
        """
        Score basé sur la durée totale (trajet + activités)
        """
        itin_duration = (
            self.get_itinerary_activity_duration(itin) +
            self.get_itinerary_travel_duration(itin)
        ) / 60.0
        itin_duration_score = np.exp(-(itin_duration - duration) ** 2)
        return itin_duration_score

    def get_lunch_time(self, itin, start_time=9):
        """
        Heure du déjeuner en fonction du premier resto rencontré
        """
        if len(itin) == 0:
            return 0.0

        itin_resto = self.get_itinerary_resto(itin, resto_cat=['Restaurants'])

        if len(itin_resto) == 0:
            return 0.0

        # On récupère les poi_id dans l'ordre de l'itinéraire
        df_itin = self.df.iloc[itin]
        poi_ids = df_itin['poi_id'].tolist()

        # Premier resto dans l'itinéraire
        first_resto_id = itin_resto[0]
        if first_resto_id not in poi_ids:
            return 0.0

        resto_itin_index = poi_ids.index(first_resto_id)

        # Durée avant le resto
        travel_duration = self.get_itinerary_travel_duration(itin[:resto_itin_index + 1]) / 60.0
        activity_duration = self.get_itinerary_activity_duration(itin[:resto_itin_index]) / 60.0

        lunch_time = start_time + travel_duration + activity_duration
        return lunch_time

    def get_itinerary_resto_score(
        self,
        itin,
        resto_cat=['Restaurants'],
        start_time=9,
        lunch_time=13
    ):
        if len(itin) == 0:
            return 0.0

        itin_resto = self.get_itinerary_resto(itin, resto_cat=resto_cat)
        resto_nbre = len(itin_resto)

        # Score sur le nombre de restos
        if resto_nbre == 0:
            resto_score = np.exp(len(itin) - 2) / np.exp(len(itin))
        else:
            resto_score = np.exp(len(itin) - resto_nbre) / np.exp(len(itin))

        # Score sur l'heure du déjeuner
        itin_lunch_time = self.get_lunch_time(itin, start_time)
        lunch_score = np.exp(-(itin_lunch_time - lunch_time) ** 2)

        return 0.7 * lunch_score + 0.3 * resto_score

    def evaluate_itinerary(
        self,
        itin,
        duration=8,
        resto_cat=['Restaurants'],
        start_time=9,
        lunch_time=13
    ):
        resto_score = self.get_itinerary_resto_score(
            itin,
            resto_cat=resto_cat,
            start_time=start_time,
            lunch_time=lunch_time
        )
        duration_score = self.get_itinerary_duration_score(itin, duration)

        return (0.6 * duration_score + 0.4 * resto_score,)

    # --------------------------------------------------------------------------
    #           Crossover / mutation
    # --------------------------------------------------------------------------

    def crossover_itinerary(self, itin1, itin2):
        itin_s = itin1 if len(itin1) < len(itin2) else itin2

        size = len(itin_s)
        if size < 2:
            return itin1, itin2

        p1, p2 = sorted(random.sample(range(size), 2))

        cr1 = itin1[p1:p2]
        cr2 = itin2[p1:p2]

        itin1[p1:p2] = cr2
        itin2[p1:p2] = cr1
        return itin1, itin2

    def mutate_itinerary(self, itin):
        if len(itin) < 2:
            return (itin,)

        p1, p2 = random.sample(range(len(itin)), 2)

        itin[p1], itin[p2] = itin[p2], itin[p1]
        return (itin,)

    # --------------------------------------------------------------------------
    #           Toolbox & génération d’itinéraires
    # --------------------------------------------------------------------------

    def generate_random_itinerary(self, itin_min_poi=5, itin_max_poi=15):
        """
        Génère un itinéraire aléatoire d’indices LOCAUX (0..n_local-1)
        """
        poi_indices = list(range(self.n_local))

        if len(poi_indices) <= itin_min_poi:
            itin_min_poi = len(poi_indices)
            itin_max_poi = len(poi_indices)

        if itin_min_poi < len(poi_indices) < itin_max_poi:
            itin_max_poi = len(poi_indices)

        itin_size = random.randint(itin_min_poi, itin_max_poi)
        itin = random.sample(poi_indices, k=itin_size)
        return creator.Itinerary(itin)

    def setup_toolbox(self, itin_min_poi=5, itin_max_poi=15):
        self.toolbox = base.Toolbox()
        self.toolbox.register("itinerary", self.generate_random_itinerary, itin_min_poi, itin_max_poi)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.itinerary)
        self.toolbox.register("evaluate", self.evaluate_itinerary)
        self.toolbox.register("mate", self.crossover_itinerary)
        self.toolbox.register("mutate", self.mutate_itinerary)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # --------------------------------------------------------------------------
    #           Boucle principale GA
    # --------------------------------------------------------------------------

    def run_ga(self, pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3):
        """Boucle principale du modèle génétique"""

        if self.n_local == 0:
            return [], 0.0

        pop = self.toolbox.population(n=pop_size)

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for itin, fit in zip(pop, fitnesses):
            itin.fitness.values = fit

        NGEN = ngen
        CXPB = cxpb
        MUTPB = mutpb

        for gen in range(NGEN):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(itin) for itin in offspring]

            for itin1, itin2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(itin1, itin2)
                    del itin1.fitness.values
                    del itin2.fitness.values

            for itin in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(itin)
                    del itin.fitness.values

            invalid_itin = [itin for itin in offspring if not itin.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_itin)
            for itin, fit in zip(invalid_itin, fitnesses):
                itin.fitness.values = fit

            pop = offspring

        best_itinerary = max(pop, key=lambda x: x.fitness.values[0])
        return list(best_itinerary), best_itinerary.fitness.values[0]