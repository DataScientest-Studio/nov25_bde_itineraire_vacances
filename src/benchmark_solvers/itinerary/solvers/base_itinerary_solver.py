class ItinerarySolverBase:
    def __init__(self, poi_df, duration_matrix, name="BaseItinerary"):
        self.df = poi_df
        self.D = duration_matrix
        self.name = name
        self.n = duration_matrix.shape[0]

    def solve(self):
        raise NotImplementedError