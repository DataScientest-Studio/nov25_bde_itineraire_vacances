from fastapi import FastAPI
import pandas as pd

import api_models as md
import database as db
import clustering as clt
import osrm
import optimizer_ga as opt


app = FastAPI()


# connexion à la base de données
dbm = db.DBManager("../data/etl_datatourisme_webservice/postgres_postgis/.env")

@app.get("/main_categories")
def get_main_catgories() :
    main_catgories_list = dbm.get_main_categories()
    return {"main_categories": main_catgories_list}

@app.post("/sub_categories")
def get_sub_categories(main_categories: md.CategoriesRequest):
    sub_categories = dbm.get_sub_categories(main_categories.categories_list)
    return {"sub_categories": sub_categories}

@app.post("/itineraries")
def get_itinerary( itin_params: md.ItineraryRequest) :
    result = dbm.search_poi(
        itin_params.longitude,
         itin_params.latitude,
         itin_params.radius,
         itin_params.sub_categories
    )
    poi_df= pd.DataFrame(result, columns =['poi_id',
                                             'longitude',
                                             'latitude',
                                             'sub_category'])
    if poi_df.shape[0] > 100 :
        poi_df = poi_df.iloc[0:100]
    
    poi_df = clt.cluster_poi(poi_df, itin_params.num_days)

    duration_matrix = osrm.get_durations_matrix(poi_df, mean = itin_params.mobility_mean)

    dict = {}

    for day in range(0, itin_params.num_days) :
        df = poi_df.loc[poi_df['day_cluster'] ==  day]
        ga = opt.GeneticAlgo(poi_df= df, 
                        duration_matrix = duration_matrix)
        ga.setup_toolbox(itin_min_poi = 5, 
                     itin_max_poi = 15)
        best_route, fitness = ga.run_ga(pop_size=50, 
                                        ngen=50, 
                                        cxpb=0.75, 
                                        mutpb=0.3)
        dict[day] = {"route": best_route, "fitness": fitness}
    
    return dict
    
        