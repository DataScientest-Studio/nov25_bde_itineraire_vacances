
import pandas as pd
import requests

osrm_url = "http://localhost:5000/table/v1"

def get_durations_matrix(poi_df, mean = 'foot') :
    #récupérer les coordonnées GPS des points :
    points = ";".join([f"{row['longitude']},{row['latitude']}" 
                   for _, row in poi_df.iterrows()])
    url =f"{osrm_url}/{mean}/{points}"
    response = requests.get(url)
    data = response.json()
    if data['code'] == 'Ok' :
        durations_matrix = pd.DataFrame(data['durations'], 
                                        index = poi_df['poi_id'].values, ## <!> l'uri n'est pas unique, besoin d'ajouter l'id d'un poi
                                        columns = poi_df['poi_id'].values).apply(lambda col : round(col/60, 1), axis = 0) # conversion des durées en minutes
        return durations_matrix    
    else :
        print(data['code'])
    
