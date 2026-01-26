from .base_itinerary_solver import ItinerarySolverBase
from benchmark_solvers.itinerary.scoring.itinerary_score import ItineraryScoring
from api.optimizer_ga import GeneticAlgo


class GAItinerarySolver(ItinerarySolverBase):

    def __init__(self, poi_df, duration_matrix,
                 pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3):

        super().__init__(poi_df, duration_matrix, name="GA_Itinerary")

        self.scoring = ItineraryScoring(poi_df, duration_matrix)

        self.ga = GeneticAlgo(poi_df, duration_matrix)
        self.ga.setup_toolbox()

        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

    def solve(self):
        itin, fitness = self.ga.run_ga(
            pop_size=self.pop_size,
            ngen=self.ngen,
            cxpb=self.cxpb,
            mutpb=self.mutpb
        )

        score = self.scoring.score(itin)
        return itin, score