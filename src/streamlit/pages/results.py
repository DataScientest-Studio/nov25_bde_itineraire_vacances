import streamlit as st
import folium as flm
from streamlit_folium import st_folium
import json
import numpy as np
import pandas as pd

from utils import fetch_main_categories, fetch_sub_categories

#------------------------------------
# import de données locales pour test
#------------------------------------
## données pour filtrer :
df_localities= pd.read_csv('data/localities.csv')
localities = df_localities['locality'].unique().tolist()


## données pour simuler les résultats 
with open("data/results_example.json", "r") as f:
    results = json.load(f)
results = results['itinerary']


categories_data = {
    "Nature & Paysages": {"icon": "tree", "color": "green"},
    "Information Touristique": {"icon": "info-circle", "color": "blue"},
    "Bien-être & Santé": {"icon": "spa", "color": "lightblue"},
    "Famille & Enfants": {"icon": "child", "color": "pink"},
    "Transports": {"icon": "car", "color": "gray"},
    "Commodités": {"icon": "shopping-basket", "color": "orange"},
    "Événements & Traditions": {"icon": "theater-masks", "color": "purple"},
    "Commerce & Shopping": {"icon": "shopping-bag", "color": "lightred"},
    "Gastronomie & Restauration": {"icon": "utensils", "color": "red"},
    "Culture & Musées": {"icon": "landmark", "color": "darkpurple"},
    "Santé & Urgences": {"icon": "hospital", "color": "darkred"},
    "Hébergement": {"icon": "hotel", "color": "beige"}, 
    "Sports & Loisirs": {"icon": "baseball", "color": "cadetblue"},
    "Services & Mobilité": {"icon": "car-side", "color": "gray"},
    "Loisirs & Clubs": {"icon": "mask", "color": "darkblue"},
    "Camping & Plein Air": {"icon": "campground", "color": "darkgreen"},
    "Patrimoine & Monuments": {"icon": "gopuram", "color": "orange"} 
}



#----------------------------------
# Sidebar pour filtrer
#------------------------------------

with st.sidebar:
    st.header("⚙️Filtres")

    # filtre par rayon :
    def update_radius():
        """ Mise à jour du rayon dans la payload"""
        st.session_state.payload['radius'] = st.session_state.radius_widget

    radius = st.slider("Rayon (km)", 
                       1, 
                       st.session_state.max_radius, 
                       value= st.session_state.payload['radius'],
                       key='radius_widget',
                       on_change= update_radius
                       )

    # filtre par nombre de jours :
    def update_num_days():
        """ Mise à jour du nombre de jours dans la payload"""
        st.session_state.payload['nb_days'] = st.session_state.num_days_widget

    num_days = st.slider("Nombre de jours", 
                         1, 
                         st.session_state.max_days, 
                         value= st.session_state.payload['nb_days'],
                         key= 'num_days_widget',
                         on_change= update_num_days)
    
    # filtre par moyen de transport :
    def update_mobility_mean():
        """ Mise à jour du moyen de transport dans la payload"""
        st.session_state.payload['osrm_mode'] = st.session_state.dict_mobility[st.session_state.mobility_mean_widget]


    index = list(st.session_state.dict_mobility.values()).index(st.session_state.payload['osrm_mode'])
    mobility_mean = st.selectbox("Moyen de mobilité/transport", 
                                 st.session_state.dict_mobility.keys(),
                                 index=index,
                                 key= 'mobility_mean_widget',
                                 on_change= update_mobility_mean)
    
    # filtre sur les catégories :
    main_categories = fetch_main_categories()

    def update_main_categories():
        """ Mise à jour des catégories principales et réinitialisation des sous-catégories dans la payload"""
    
        st.session_state.payload['main_categories'] = st.session_state.main_cat_widget
        st.session_state.payload['sub_categories'] = []

    main_cat = st.multiselect("Catégorie(s) principale(s)", 
                              main_categories, 
                              default = st.session_state.payload['main_categories'],
                              key = 'main_cat_widget',
                              on_change=update_main_categories)
    
    if main_cat :
        sub_categories = fetch_sub_categories(main_cat)
    else :
        sub_categories = []

    def update_sub_categories():
        """fonction pour mettre à jour les sub_categories dans la payload au 
        au changement du champ correspondant"""
        st.session_state.payload['sub_categories'] = st.session_state.sub_cat_widget

    sub_cat = st.multiselect("Sous-catégorie(s)", 
                             sub_categories , 
                             default = st.session_state.payload['sub_categories'],
                             key='sub_cat_widget',
                             on_change= update_sub_categories)
    

    if st.button('Mettre à jour') :
        payload = st.session_state.payload
        if (payload["main_categories"]== [] ) or (payload["sub_categories"] == []) or (payload["nb_days"]== 0) or (payload["osrm_mode"] == "") or (payload["radius"] == 0):
            st.error("❌ Un ou plusieurs paramètres de filtres sont invalides") 
        else : 
            st.write(st.session_state.payload)




#------------------------------------
# Partie centrale avec les résultats
#------------------------------------
st.header("🗺️ Nos propositions d'itinéraires")
col1, col2 = st.columns(2)

for day in range(0, len(results)) :
    itinerary = results[day]['pois']

    #création de la carte pour la journée :
    ## détérmination du point central de la carte par rapport à l'itinéraire :
    central_lt = np.mean([itinerary[i]['latitude'] for i in range(0,len(itinerary))])
    central_lg = np.mean([itinerary[i]['longitude'] for i in range(0,len(itinerary))])

    ## Création de la carte  :
    m = flm.Map(location=[central_lt, central_lg],
                zoom_start=12, 
                width=325, 
                height=300)

    
    ## Ajout des pois de l'itinéraire  :
    for i in range(0, len(itinerary)) :
        poi_lt = itinerary[i]['latitude']
        poi_lg = itinerary[i]['longitude']
        poi_main_cat = itinerary[i]['main_category']
        
        flm.Marker(location = [poi_lt, poi_lg],
                   popup=f"<b>{poi_main_cat}</b>",
                   icon = flm.Icon(prefix= 'fa',
                                   icon= categories_data[poi_main_cat]['icon'],
                                   color= categories_data[poi_main_cat]['color'],
                                   icon_color='white'
                                   )
                   ).add_to(m)

    if day%2 == 0 :
        with col1 :      
            st.subheader(f"Journée n°{day+1}")
            st_folium(m, width=325, height=300)
    else: 
        with col2 :      
            st.subheader(f"Journée n°{day+1}")
            st_folium(m, width=325, height=300)

    
