import pandas as pd
from fastapi import FastAPI

from src.api import api_models as md
from src.api import clustering as clt
from src.api import database as db
from src.api import optimizer_ga as opt
from src.api import osrm

app = FastAPI()


# connexion à la base de données
dbm = db.DBManager()

@app.get("/main_categories")
def get_main_catgories():
    main_catgories_list = dbm.get_main_categories()
    return {"main_categories": main_catgories_list}


@app.post("/sub_categories")
def get_sub_categories(main_categories: md.CategoriesRequest):
    sub_categories = dbm.get_sub_categories(main_categories.categories_list)
    return {"sub_categories": sub_categories}


@app.post("/itineraries")
def get_itinerary(itin_params: md.ItineraryRequest):
    result = dbm.search_poi(
        itin_params.longitude,
        itin_params.latitude,
        itin_params.radius,
        itin_params.sub_categories,
    )
    poi_df = pd.DataFrame(
        result, columns=["poi_id", "longitude", "latitude", "sub_category"]
    )
    if poi_df.shape[0] > 100:
        poi_df = poi_df.iloc[0:100]

    poi_df = clt.cluster_poi(poi_df, itin_params.num_days)

    poi_gps = poi_df.drop(columns="sub_category").drop_duplicates()
    duration_matrix = osrm.get_durations_matrix(poi_gps, mean=itin_params.mobility_mean)

    dict = {}

    for day in range(0, itin_params.num_days):
        df = poi_df.loc[poi_df["day_cluster"] == day]
        ga = opt.GeneticAlgo(poi_df=df, duration_matrix=duration_matrix)
        ga.setup_toolbox(itin_min_poi=5, itin_max_poi=15)
        best_route, fitness = ga.run_ga(pop_size=50, ngen=50, cxpb=0.75, mutpb=0.3)

        # récupération des données descriptives de chaque poi de l'itinéraire :
        best_route_data = {poi_id: dbm.get_poi_data(poi_id) for poi_id in best_route}

        # récupération du tracé de l'itinéraire sur osrm :
        best_route_gps = poi_gps.loc[poi_gps["poi_id"].isin(best_route)]
        best_route_line = osrm.get_itin_route(
            best_route_gps, mean=itin_params.mobility_mean
        )
        best_route_line = [
            [lt, lg] for lt, lg in best_route_line
        ]  # coordonnées dans le format (lat, long) pour Folium

        dict[day] = {
            "route_score": fitness,
            "route_data": best_route_data,
            "route_line": best_route_line,
        }

    return dict
