
import pandas as pd
import requests
import polyline

osrm_url_table = "http://localhost:5000/table/v1"

def get_durations_matrix(poi_df, mean = 'foot') :
    #récupérer les coordonnées GPS des points :
    points = ";".join([f"{row['longitude']},{row['latitude']}" 
                   for _, row in poi_df.iterrows()])
    url =f"{osrm_url_table}/{mean}/{points}"
    response = requests.get(url)
    data = response.json()

    if data['code'] == 'Ok' :
        durations_matrix = pd.DataFrame(data['durations'], 
                                        index = poi_df['poi_id'].values, ## <!> l'uri n'est pas unique, besoin d'ajouter l'id d'un poi
                                        columns = poi_df['poi_id'].values).apply(lambda col : round(col/60, 1), axis = 0) # conversion des durées en minutes
        return durations_matrix    
    else :
        print(data['code'])
    

osrm_url_route = "http://localhost:5000/route/v1"

def get_itin_route(route_gps, mean = 'foot') :
    #récupérer les coordonnées GPS des points :
    points = ";".join([f"{row['longitude']},{row['latitude']}" 
                   for _, row in route_gps.iterrows()])
    url =f"{osrm_url_route}/{mean}/{points}"
    response = requests.get(url)
    data = response.json()

    if data['code'] == 'Ok' :
        route_geometry = data['routes'][0]['geometry'] # récupération de la route sous forme de polyline
        decoded_route = polyline.decode(route_geometry) # décodage de la polyline pour récupérer lat, long de la route
                                        
        return decoded_route    
    else :
        print(data['code'])