import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium as flm
from streamlit_folium import st_folium




st.title("Optimisateur d'itinéraire de vacances")
# st.markdown("application pour planifier vos vacances en toute tranquilité")
st.header("Choix des paramètres")


st.write("saisissez les paramètres de vos vacances")

# catégories principales
main_cat_url = "http://localhost:8000/main_categories"
try :
    response = requests.get(main_cat_url)
    response.raise_for_status() 
    data = response.json()
    selected_main_cat = st.multiselect("Catégorie(s) principale(s)", data['main_categories'], default = [])

except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération des données: {e}")



# catégories secondaires :
sub_cat_url = "http://localhost:8000/sub_categories"
if selected_main_cat :
    try :
        params= {"categories_list": selected_main_cat}
        response = requests.post(sub_cat_url, json= params)
        response.raise_for_status() 
        data = response.json()
        selected_sub_cat = st.multiselect("sous-catégorie(s)", data['sub_categories'], default = [])
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
else :
    st.multiselect("sous-catégorie(s)", [])

#autres inputs de l'utilisateur :
num_days = st.slider("Nombre de jour", 1, 10)
longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
radius = st.slider("Nombre de Km", 1, 30)*1000
mobility_mean = st.selectbox("Moyen de transport", ['à pied', 'en voiture'])

dict_mobility = {'à pied' : 'foot', 'en voiture': 'car'}

mobility_mean = dict_mobility[mobility_mean]

if st.button('Proposer des itinéraires') :
    if not selected_sub_cat or longitude == 0 or latitude == 0 :
        st.error("merci de remplir le formulaire des paramètres ")
    else : 
        itin_dict = {
            'sub_categories' : selected_sub_cat,
            'longitude' : longitude,
            'latitude' : latitude,
            'radius' : radius,
            'num_days' : num_days,
            'mobility_mean' : mobility_mean
             }

    itin_url = "http://localhost:8000/itineraries"
    response = requests.post(itin_url, json= itin_dict)
    response.raise_for_status()
    
    # sauvegarde des résultats :
    st.session_state.itineraries = response.json()
    st.session_state.num_days = num_days

if 'itineraries' in st.session_state:
    st.header("Résultats")
    data = st.session_state.itineraries
    for day in range(0, st.session_state.num_days) :
        route_score = data[f"{day}"]['route_score']
        route_data = data[f"{day}"]['route_data']
        route_line = data[f"{day}"]['route_line']
        
        #création de la carte pour la journée :
        ## détérmination du point central de la carte par rapport à l'itinéraire :
        central_lt = np.mean([route_data[id][3] for id in route_data.keys()])
        central_lg =np.mean([route_data[id][4] for id in route_data.keys()])

        ## Création de la carte  :
        m = flm.Map(location=[central_lt, central_lg],
                    zoom_start=10, 
                    width=750, 
                    height=500)
        
        ## Ajout des pois de l'itinéraire  :
        for poi_id, poi_data in route_data.items() :
            poi_lt = poi_data[3]
            poi_lg = poi_data[4]
            poi_name = poi_data[1]
            poi_description = poi_data[2]
        
            flm.Marker(location = [poi_lt, poi_lg],
                       popup = poi_description,
                       tooltip = poi_name,
                       icon = flm.Icon(color= 'purple')
                        ).add_to(m)
        
        ## Ajout de l'itinéraire :
        flm.PolyLine(route_line, 
                     color= 'red',
                     weight=4,
                     opacity=0.8
                     ).add_to(m)


        st.subheader(f"Itinéraire de la journée n°{day+1}")
        st_folium(m, width=750, height=500)

