from deap import base
from deap import creator
from deap import tools

import random
import numpy as np

        


class GeneticAlgo() :

    def __init__(self, poi_df, duration_matrix) :

        # Supprimer les classes si elles existent déjà
        if hasattr(creator, "FitnessItinerary"):
            del creator.FitnessItinerary
        if hasattr(creator, "Itinerary"):
            del creator.Itinerary

        # intitialisation de la classs fitness = fonction de sélection :
        creator.create("FitnessItinerary", base.Fitness, weights=(1.0,)) # weights positif car maximisation

        # intiialisation de la classe individu = un itinéraire :
        creator.create("Itinerary", list, fitness = creator.FitnessItinerary)

        # création de la toolbox (conteneur de toutes les opérations):
        self.toolbox = base.Toolbox()
        self.df = poi_df
        self.matrix = duration_matrix
    
    ##------------------------------------------------------------------------------
    ###           fonction de sélection (fitness function)
    ##------------------------------------------------------------------------------

    
    # définition de fonctions qui calculent la durée d'un itinéraire : durée du trajet /durée des activités :
    def get_itinerary_travel_duration(self, itin) :
        """
        calul la durée de voyage pour un itinéraire de pois en minutes
        arguement :
            itin : liste des id des pois d'un itinéraire
            duration_matrix : matrice des durées entre les pois couvrant les pois de l'itinéraire
        """
        travel_duration = sum([self.matrix[itin[i]][itin[i+1]] for i in range(len(itin)-1)])
        return travel_duration

    # défintion d'une foction pour évaluer le score d'un itinéraire sur la base de la restauration de l'itinéraire  :
    def get_itinerary_resto(self, itin, resto_cat = ['Restaurants']):
        """
        retourne les pois qui sont dans l'une des sub_categories resto_cat

        arguement :
            itin : liste des id des pois d'un itinéraire
            df : dataframe avec les données des pois
            resto_cat : sous catégories à considérer pour la restauration
        """
        df_itin = self.df.loc[self.df.poi_id.isin(itin)]
        itin_resto = df_itin.loc[df_itin.sub_category.isin(resto_cat), 'poi_id']
        return itin_resto

    def get_itinerary_activity_duration(self, itin, lunch_duration = 60, activity_duration= 45, resto_cat = ['Restaurants']) :
        """
        calul la durée des activités pour un itinéraire de pois en minutes
        arguement :
            itin : liste des id des pois d'un itinéraire
            df : dataframe avec les données des pois
            lunch_duration : durée du repas de midi, par défaut 60 min
            activity_duration : durée d'une activité, hors repas, par défaut 45 min
        """
        itin_resto = self.get_itinerary_resto(itin, resto_cat= resto_cat)

        activities_duration= sum([lunch_duration if poi in itin_resto
                                                else activity_duration 
                                                for poi in itin] )
        return activities_duration    

    # défintion d'une fonction pour évaluer le score d'un itinéraire sur la base de sa durée   :
    def get_itinerary_duration_score(self, itin, duration = 8) :
        """
        calule le score d'un itinéraire sur la base d'une durée moyenne de la journée
        arguement :
            itin : liste des id des pois d'un itinéraire
            duration : durée moyenne d'une journée à réaliser l'itinéraire, par défaut 8 heures
        """
        itin_duration = (self.get_itinerary_activity_duration(itin) + self.get_itinerary_travel_duration(itin))/60
        itin_duration_score = np.exp(-(itin_duration - duration )**2) # en peut modifier l'intervalle de tolérance en multipliant ce qui est à l'intérieur
                                                                    # de l'exp.
        return itin_duration_score

    def get_lunch_time(self, itin, start_time = 9):
        itin_resto = self.get_itinerary_resto(itin, resto_cat = ['Restaurants'])
        
        if itin_resto.shape[0] == 0 : 
            return 0
        else :         
            resto_itin_index = itin.index(itin_resto.iloc[0]) # on récupère son index dans l'itinéraire
            # durée itinéraire avant le retaurant :
            travel_duration = self.get_itinerary_travel_duration(itin[ : resto_itin_index+1])/60 
            activity_duration =  self.get_itinerary_activity_duration(itin[ : resto_itin_index])/60
            lunch_time = start_time + travel_duration + activity_duration
            return lunch_time

    def get_itinerary_resto_score(self, itin, resto_cat = ['Restaurants'] , start_time = 9, lunch_time = 13) :
        # sléection des pois restaurant et comptage de leur nombre : 
        itin_resto = self.get_itinerary_resto(itin, resto_cat = resto_cat)
        resto_nbre = itin_resto.shape[0]
        
        # calucl d'un score en fonction du nombre de restaurant :
        if resto_nbre == 0 :
            resto_score = np.exp(len(itin) - 2) / np.exp(len(itin)) # même score de 0 retaurant est le même que le score du nbre de resto = 2
        else:
            resto_score = np.exp(len(itin) - resto_nbre ) / np.exp(len(itin))
        
        # calcul du score du score sur le lunchtime :
        itin_lunch_time = self.get_lunch_time(itin, start_time)
        lunch_score = np.exp(-(itin_lunch_time - lunch_time)**2)
        
        return 0.7* lunch_score + 0.3 * resto_score

    # évalution de l'initnéraire : fitness function
    def evaluate_itinerary(self, itin, duration = 8, resto_cat = ['Restaurants'], start_time = 9, lunch_time = 13):
        
        resto_score = self.get_itinerary_resto_score(itin, resto_cat = resto_cat, start_time = start_time, lunch_time = lunch_time)
        duration_score = self.get_itinerary_duration_score(itin, duration)

        return (0.6 * duration_score + 0.4 * resto_score ,)

    ##------------------------------------------------------------------------------
    ###           Fonction de reproduction et de mutation
    ##------------------------------------------------------------------------------

    def crossover_itinerary(self, itin1, itin2):
        # identification du plus petit (small) des itinéraires
        itin_s = itin1 if len(itin1) < len(itin2) else itin2

        # choix des points de coupure sur la longueur en commun :
        size=  len(itin_s)
        p1, p2 = sorted(random.sample(range(size), 2)) 

        # la section à échanger :
        cr1 = itin1[p1:p2]
        cr2 = itin2[p1:p2]
        # échange des sections 
        itin1[p1:p2] = cr2
        itin2[p1:p2] = cr1
        return itin1, itin2

    def mutate_itinerary(self, itin):
        # choisir deux points aléatoire dans l'itinéraire :
        p1, p2 = random.sample(range(len(itin)), 2)
                        
        # identification des poi et changement d'ordre :
        poi1 = itin[p1]
        poi2 = itin[p2]

        itin[p1] = poi2
        itin[p2] = poi1

        return (itin,)
    
    ##--------------------------------------------------------------------------------
    ###       création et configuration de la toolbox : conteneurs des opérations GA
    ##--------------------------------------------------------------------------------

    def setup_toolbox(self, itin_min_poi = 5, itin_max_poi = 15) :
        # création de la toolbox (conteneur de toutes les opérations):
        self.toolbox = base.Toolbox()

        def generate_random_itinerary(itin_min_poi = 5, itin_max_poi = 15) :
            """ génère aléatoirement des itinéraires aléatoires d'une longueur comprise entre un min et un max 
            à partir d'une liste d'index de poi 
            arguments :
            - poi_list : list d'identifiant de pois
            - itin_min_poi : longueur minimale d'un itinéraire
            - itin_max_poi : longueur maximale d'un itinéraire
            """
            poi_list = self.df['poi_id'].tolist()

            if len(poi_list) <= itin_min_poi :
                itin_min_poi = len(poi_list)
                itin_max_poi =  len(poi_list)
            
            if itin_min_poi < len(poi_list) < itin_max_poi :
                itin_max_poi =  len(poi_list)

            itin_size = random.randint(itin_min_poi, itin_max_poi)
            itin = random.sample(poi_list, k= itin_size)
            return creator.Itinerary(itin)

        self.toolbox.register("itinerary", generate_random_itinerary)
          
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.itinerary)
           
        self.toolbox.register("evaluate", self.evaluate_itinerary)
                                
        self.toolbox.register("mate", self.crossover_itinerary)

        self.toolbox.register("mutate", self.mutate_itinerary)

        self.toolbox.register("select", tools.selTournament, tournsize=3) # comparaions de 3 itinéraires à la fois

    ##--------------------------------------------------------------------------------
    ###       algorithme génétique : méthode principale
    ##--------------------------------------------------------------------------------

    def run_ga(self, pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3 ):
        """ Loop principale du modèle génétique"""
        
        # création de la population des itinéraires
        pop = self.toolbox.population(n= pop_size)  # 50 itinéraires
        
        # évaluation initiale de la population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for itin, fit in zip(pop, fitnesses):
            itin.fitness.values = fit  # affecter la valeur de la fiteness à chaque indiv
        
        # paramètre de l'algo :
        NGEN = ngen      # nombre de générations
        CXPB = cxpb     # probabilité de crossover : prendre 70% de la population de chaque génération
        MUTPB = mutpb   # probabilité de mutation : 20% de chance de mutation pour chaque individu

        # Application de l'évolution :
        for gen in range(NGEN):
            #print(f"Generation {gen}")
            
            # Sélection des itinéraires pour la reproduction :
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(itin) for itin in offspring]
            
            # Reproduction :
            for itin1, itin2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(itin1, itin2)
                    # Important: supression de la fiteness de validation
                    del itin1.fitness.values
                    del itin2.fitness.values
            
            # Mutation : 
            for itin in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(itin)
                    del itin.fitness.values
            
            # evaluation de la fitness des individus sans fitness
            invalid_itin = [itin for itin in offspring if not itin.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_itin)
            for itin, fit in zip(invalid_itin, fitnesses):
                itin.fitness.values = fit
            
            # Replacement de la population initial avec la nouvelles génération:
            pop = offspring
            
            # Meilleurs score de la génération obtenue :
            #best = max(pop, key=lambda x: x.fitness.values[0])
            #print(f" meilleur score de la génération {gen}: {best.fitness.values[0]:.4f}")
        
        # meilleurs itinéraire  : 
        best_itinerary = max(pop, key=lambda x: x.fitness.values[0])
        return best_itinerary, best_itinerary.fitness.values[0]
