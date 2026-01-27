import numpy as np


class ItineraryScoring:

    def __init__(self, poi_df, matrix):
        self.df = poi_df
        self.matrix = matrix

    def travel_duration(self, itin):
        return sum(self.matrix[itin[i], itin[i+1]] for i in range(len(itin)-1))

    def activity_duration(self, itin, lunch_duration=60, activity_duration=45):
        df_itin = self.df.loc[self.df.osrm_index.isin(itin)]
        resto = df_itin.loc[df_itin.sub_category == "Restaurants", "osrm_index"].tolist()

        return sum(lunch_duration if poi in resto else activity_duration for poi in itin)

    def lunch_time(self, itin, start_time=9):
        df_itin = self.df.loc[self.df.osrm_index.isin(itin)]
        resto = df_itin.loc[df_itin.sub_category == "Restaurants", "osrm_index"].tolist()

        if not resto:
            return None

        idx = itin.index(resto[0])
        travel = self.travel_duration(itin[:idx+1]) / 60
        activity = self.activity_duration(itin[:idx]) / 60

        return start_time + travel + activity

    def score(self, itin):
        duration = (self.activity_duration(itin) + self.travel_duration(itin)) / 60
        duration_score = np.exp(-(duration - 8)**2)

        lunch = self.lunch_time(itin)
        lunch_score = np.exp(-(lunch - 13)**2) if lunch else 0.1

        return 0.6 * duration_score + 0.4 * lunch_score