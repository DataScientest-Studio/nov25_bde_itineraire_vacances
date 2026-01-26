import numpy as np


def itinerary_metrics(scores, times, routes, poi_df, matrix):
    if len(scores) == 0:
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "best_score": 0.0,
            "mean_time": 0.0,
            "variety": 0,
            "best_route": [],
        }

    best_idx = int(np.argmax(scores))
    best_route = routes[best_idx]
    best_score = float(scores[best_idx])

    df_itin = poi_df.loc[poi_df.osrm_index.isin(best_route)]
    variety = int(df_itin.sub_category.nunique())

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "best_score": best_score,
        "mean_time": float(np.mean(times)),
        "variety": variety,
        "best_route": best_route,
    }