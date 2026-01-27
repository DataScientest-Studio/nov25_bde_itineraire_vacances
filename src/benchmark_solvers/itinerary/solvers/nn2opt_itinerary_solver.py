from .base_itinerary_solver import ItinerarySolverBase
from benchmark_solvers.itinerary.scoring.itinerary_score import ItineraryScoring

class NN2OptItinerarySolver(ItinerarySolverBase):

    def __init__(self, poi_df, duration_matrix, start=0):
        super().__init__(poi_df, duration_matrix, name="NN2Opt_Itinerary")
        self.start = start
        self.scoring = ItineraryScoring(poi_df, duration_matrix)

    def nearest_neighbor(self):
        unvisited = set(range(self.n))
        unvisited.remove(self.start)

        route = [self.start]
        current = self.start

        while unvisited:
            next_node = min(unvisited, key=lambda j: self.D[current, j])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return route

    def two_opt(self, route):
        improved = True
        best_route = route
        best_score = self.scoring.score(route)

        while improved:
            improved = False
            for i in range(1, self.n - 2):
                for k in range(i + 1, self.n - 1):
                    new_route = (
                        best_route[:i]
                        + best_route[i:k+1][::-1]
                        + best_route[k+1:]
                    )
                    new_score = self.scoring.score(new_route)

                    if new_score > best_score:  # maximisation
                        best_route = new_route
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def solve(self):
        route = self.nearest_neighbor()
        route = self.two_opt(route)
        score = self.scoring.score(route)
        return route, score