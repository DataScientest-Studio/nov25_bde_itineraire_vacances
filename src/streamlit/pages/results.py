import json

import folium as flm
import folium
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
from utils import (
    distance_print,
    fetch_main_categories,
    fetch_sub_categories,
    send_payload,
    time_print,
    get_selected_pois
)

import streamlit as st

#------------------------------------
# Données pour la visualisation 
#------------------------------------

COLORS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "cadetblue", "darkgreen", "pink"
]


categories_data = {
    "Restauration rapide": {"icon": "hamburger", "color": "red"},
    "Châteaux & Fortifications": {"icon": "fort-awesome", "color": "orange"},
    "Religieux": {"icon": "church", "color": "darkpurple"},
    "Côtes & littoral": {"icon": "water", "color": "blue"},
    "Sports de balle & raquette": {"icon": "table-tennis", "color": "cadetblue"},
    "Artisanat": {"icon": "palette", "color": "orange"},
    "Eau & Milieux humides": {"icon": "tint", "color": "blue"},
    "Commerces": {"icon": "store", "color": "orange"},
    "Bibliothèques & médiation": {"icon": "book", "color": "darkpurple"},
    "Loisirs indoor": {"icon": "gamepad", "color": "purple"},
    "Producteurs": {"icon": "seedling", "color": "green"},
    "Antiquités & brocante": {"icon": "archive", "color": "beige"},
    "Restaurants": {"icon": "utensils", "color": "red"},
    "Marchés": {"icon": "shopping-basket", "color": "orange"},
    "Parcs & loisirs": {"icon": "seedling", "color": "green"},
    "Éducation & apprentissage": {"icon": "graduation-cap", "color": "blue"},
    "Forêts & milieux naturels": {"icon": "tree", "color": "darkgreen"},
    "Aire de pique-nique": {"icon": "picnic", "color": "green"},
    "Sports équestres": {"icon": "horse", "color": "beige"},
    "Trains & bus touristiques": {"icon": "train", "color": "blue"},
    "Zoo & animaux": {"icon": "paw", "color": "green"},
    "Spectacle vivant": {"icon": "theater-masks", "color": "purple"},
    "Bars & cafés": {"icon": "coffee", "color": "darkred"},
    "Rencontres & conférences": {"icon": "comments", "color": "blue"},
    "Sports nautiques": {"icon": "swimmer", "color": "lightblue"},
    "Paysages remarquables": {"icon": "mountain", "color": "green"},
    "Montagne & Relief": {"icon": "mountain", "color": "gray"},
    "Ouvrages d'art": {"icon": "bridge", "color": "gray"},
    "Produits locaux": {"icon": "leaf", "color": "green"},
    "Antiquité & Vestiges": {"icon": "columns", "color": "beige"},
    "Golf & mini-golf": {"icon": "golf-ball", "color": "green"},
    "Musées & expositions": {"icon": "landmark", "color": "darkpurple"},
    "Téléphériques & remontées": {"icon": "cable-car", "color": "blue"},
    "Eau vive & cascades": {"icon": "water", "color": "lightblue"},
    "Aires & jeux": {"icon": "child", "color": "pink"},
    "Patrimoine rural & agricole": {"icon": "tractor", "color": "green"},
    "Thermalisme": {"icon": "hot-tub", "color": "lightblue"},
    "Sports collectifs & stades": {"icon": "futbol", "color": "cadetblue"},
    "Cinéma & audiovisuel": {"icon": "film", "color": "darkpurple"},
    "Jeune public": {"icon": "baby", "color": "pink"},
    "Géologie & curiosités": {"icon": "gem", "color": "gray"},
    "Sports mécaniques": {"icon": "motorcycle", "color": "darkred"},
    "Patrimoine civil": {"icon": "building", "color": "gray"},
    "Sports outdoor": {"icon": "hiking", "color": "green"},
    "Concerts & musique": {"icon": "music", "color": "purple"},
    "Fêtes & traditions": {"icon": "gifts", "color": "purple"},
    "Festivals & grands événements": {"icon": "star", "color": "purple"},
    "Soins & bien-être": {"icon": "spa", "color": "lightblue"},
    "Foires & salons": {"icon": "users", "color": "orange"},
    "Cimetières & mémoriaux": {"icon": "monument", "color": "gray"},
    "Glace & haute montagne": {"icon": "icicles", "color": "lightblue"},
    "Sports d'hiver": {"icon": "skiing", "color": "blue"},
    "Thalasso & balnéo": {"icon": "swimming-pool", "color": "lightblue"},
    "Aventure & accrobranche": {"icon": "tree", "color": "darkgreen"},
    "Défilés & parades": {"icon": "flag", "color": "purple"},
    "Vins & spiritueux": {"icon": "wine-glass", "color": "darkred"},
}

categories_emoji = {
    "Restauration rapide": "🍔",
    "Châteaux & Fortifications": "🏰",
    "Religieux": "⛪",
    "Côtes & littoral": "🌊",
    "Sports de balle & raquette": "🎾",
    "Artisanat": "🎨",
    "Eau & Milieux humides": "💧",
    "Commerces": "🏪",
    "Bibliothèques & médiation": "📚",
    "Loisirs indoor": "🎮",
    "Producteurs": "🌾",
    "Antiquités & brocante": "🗃️",
    "Restaurants": "🍽️",
    "Marchés": "🛒",
    "Parcs & loisirs": "🌳",
    "Éducation & apprentissage": "🎓",
    "Forêts & milieux naturels": "🌲",
    "Aire de pique-nique": "🧺",
    "Sports équestres": "🐴",
    "Trains & bus touristiques": "🚂",
    "Zoo & animaux": "🐾",
    "Spectacle vivant": "🎭",
    "Bars & cafés": "☕",
    "Rencontres & conférences": "💬",
    "Sports nautiques": "🏊",
    "Paysages remarquables": "🏔️",
    "Montagne & Relief": "⛰️",
    "Ouvrages d'art": "🌉",
    "Produits locaux": "🍃",
    "Antiquité & Vestiges": "🏛️",
    "Golf & mini-golf": "⛳",
    "Musées & expositions": "🖼️",
    "Téléphériques & remontées": "🚡",
    "Eau vive & cascades": "💦",
    "Aires & jeux": "🎪",
    "Patrimoine rural & agricole": "🚜",
    "Thermalisme": "♨️",
    "Sports collectifs & stades": "⚽",
    "Cinéma & audiovisuel": "🎬",
    "Jeune public": "👶",
    "Géologie & curiosités": "💎",
    "Sports mécaniques": "🏍️",
    "Patrimoine civil": "🏢",
    "Sports outdoor": "🥾",
    "Concerts & musique": "🎵",
    "Fêtes & traditions": "🎁",
    "Festivals & grands événements": "⭐",
    "Soins & bien-être": "🧘",
    "Foires & salons": "👥",
    "Cimetières & mémoriaux": "🗿",
    "Glace & haute montagne": "🧊",
    "Sports d'hiver": "⛷️",
    "Thalasso & balnéo": "🏊‍♂️",
    "Aventure & accrobranche": "🌲",
    "Défilés & parades": "🚩",
    "Vins & spiritueux": "🍷",
}

categories_color = {
    "Restauration rapide": "red",
    "Châteaux & Fortifications": "orange",
    "Religieux": "violet",
    "Côtes & littoral": "blue",
    "Sports de balle & raquette": "blue",
    "Artisanat": "orange",
    "Eau & Milieux humides": "blue",
    "Commerces": "orange",
    "Bibliothèques & médiation": "violet",
    "Loisirs indoor": "violet",
    "Producteurs": "green",
    "Antiquités & brocante": "orange",
    "Restaurants": "red",
    "Marchés": "orange",
    "Parcs & loisirs": "green",
    "Éducation & apprentissage": "blue",
    "Forêts & milieux naturels": "green",
    "Aire de pique-nique": "green",
    "Sports équestres": "orange",
    "Trains & bus touristiques": "blue",
    "Zoo & animaux": "green",
    "Spectacle vivant": "violet",
    "Bars & cafés": "red",
    "Rencontres & conférences": "blue",
    "Sports nautiques": "blue",
    "Paysages remarquables": "green",
    "Montagne & Relief": "gray",
    "Ouvrages d'art": "gray",
    "Produits locaux": "green",
    "Antiquité & Vestiges": "orange",
    "Golf & mini-golf": "green",
    "Musées & expositions": "violet",
    "Téléphériques & remontées": "blue",
    "Eau vive & cascades": "blue",
    "Aires & jeux": "violet",
    "Patrimoine rural & agricole": "green",
    "Thermalisme": "blue",
    "Sports collectifs & stades": "blue",
    "Cinéma & audiovisuel": "violet",
    "Jeune public": "violet",
    "Géologie & curiosités": "gray",
    "Sports mécaniques": "red",
    "Patrimoine civil": "gray",
    "Sports outdoor": "green",
    "Concerts & musique": "violet",
    "Fêtes & traditions": "violet",
    "Festivals & grands événements": "violet",
    "Soins & bien-être": "blue",
    "Foires & salons": "orange",
    "Cimetières & mémoriaux": "gray",
    "Glace & haute montagne": "blue",
    "Sports d'hiver": "blue",
    "Thalasso & balnéo": "blue",
    "Aventure & accrobranche": "green",
    "Défilés & parades": "violet",
    "Vins & spiritueux": "red",
}


# ----------------------------------
# Sidebar pour filtrer
# ------------------------------------

with st.sidebar:
    st.header("⚙️Filtres")

    # filtre par rayon :
    def update_radius():
        """Mise à jour du rayon dans la payload"""
        st.session_state.payload["radius"] = st.session_state.radius_widget

    radius = st.slider(
        "Rayon (km)",
        1,
        st.session_state.max_radius,
        value=st.session_state.payload["radius"],
        key="radius_widget",
        on_change=update_radius,
    )

    # filtre par nombre de jours :
    def update_num_days():
        """Mise à jour du nombre de jours dans la payload"""
        st.session_state.payload["days"] = st.session_state.num_days_widget


    num_days = st.slider(
        "Nombre de jours",
        1,
        st.session_state.max_days,
        value=st.session_state.payload["days"],
        key="num_days_widget",
        on_change=update_num_days,
    )

    # filtre par moyen de transport :
    def update_mobility_mean():

        """Mise à jour du moyen de transport dans la payload"""
        st.session_state.payload["transport_mode"] = st.session_state.dict_mobility[
            st.session_state.mobility_mean_widget
        ]

    index = list(st.session_state.dict_mobility.values()).index(
        st.session_state.payload["transport_mode"]
    )
    mobility_mean = st.selectbox(
        "Moyen de mobilité/transport",
        st.session_state.dict_mobility.keys(),
        index=index,
        key="mobility_mean_widget",
        on_change=update_mobility_mean,
    )

    # filtre sur les catégories :
    main_categories = fetch_main_categories()

    def update_main_categories():
        """Mise à jour des catégories principales et réinitialisation des sous-catégories dans la payload"""

        st.session_state.payload["main_category"] = st.session_state.main_cat_widget


    main_cat = st.multiselect(
        "Catégorie(s) principale(s)",
        main_categories,
        default=st.session_state.payload["main_category"],
        key="main_cat_widget",
        on_change=update_main_categories,
    )

    if main_cat:
        sub_categories = fetch_sub_categories(main_cat)
        # Filtrer les sous-catégories invalides
        valid_sub_cat = [
            cat for cat in st.session_state.payload["sub_category"] 
            if cat in sub_categories
        ]
    else:
        sub_categories = []
        valid_sub_cat = []

    def update_sub_categories():
        """fonction pour mettre à jour les sub_categories dans la payload au
        au changement du champ correspondant"""
        st.session_state.payload["sub_category"] = st.session_state.sub_cat_widget


    sub_cat = st.multiselect(
        "Sous-catégorie(s)",
        sub_categories,
        default=valid_sub_cat,
        key="sub_cat_widget",
        on_change=update_sub_categories,
    )

    if st.button("Mettre à jour", type="primary"):
        payload = st.session_state.payload
        if (
            (payload["main_category"] == [])
            or (payload["sub_category"] == [])
            or (payload["days"] == 0)
            or (payload["transport_mode"] == "")
            or (payload["radius"] == 0)
        ):
            ## DEBUG
            #st.write(payload)
            st.error("❌ Un ou plusieurs paramètres de filtres sont invalides")
        else:
            pois = get_selected_pois(payload)

            # Construction du payload pour /itinerary/compute
            itinerary_payload = {
                "pois": pois["pois"],
                "days": payload["days"],
                "transport_mode": payload["transport_mode"],
                "solver": payload["solver"],
                "latitude": payload["latitude"],
                "longitude": payload["longitude"],
            }
            # Sauvegarde dans la session
            st.session_state.itinerary_payload = itinerary_payload

                        
            # Supprimer le cache pour forcer le recalcul
            if "itinerary_result" in st.session_state:
               del st.session_state.itinerary_result
            
            st.rerun()


#-------------------------------------------------------
#   Récupération et affichage des itinéraires
#-------------------------------------------------------

## DEBUG
#st.write(st.session_state.itinerary_payload)

# calcul itinéraire s'il n'existe pas : 
if "itinerary_result" not in st.session_state:
    with st.spinner("Calcul de l'itinéraire en cours..."):
        st.session_state.itinerary_result = send_payload(st.session_state.itinerary_payload)


itinerary = st.session_state.itinerary_result
itinerary_list = itinerary["itinerary"]

## DEBUG
# Affichage de la liste brute
st.write("Itinerary list :", itinerary_list)


for day in range(0, len(itinerary_list)):
    # récupération de l'itinéraire :
    itinerary = itinerary_list[day]["pois"]
    #st.write(f"Jour {day + 1} :", itinerary)



#------------------------------------
#    Partie centrale avec les résultats
# ------------------------------------

#st.header("🗺️ Nos propositions d'itinéraires")

if "show_details" not in st.session_state:
    st.session_state.show_details = {}

# Récupération de l'icône du mode principale de mobilité :

index = list(st.session_state.dict_mobility.values()).index(
    st.session_state.payload["transport_mode"]
)
mobility_mode_icon = list(st.session_state.dict_mobility.keys())[index][0]


# Affichage de résultat de chaque jour : 
for day in range(0, len(itinerary_list)):
    
    # récupération de l'itinéraire :
    itinerary = itinerary_list[day]["pois"]


    # création de la carte pour la journée :
    ## détérmination du point central de la carte par rapport à l'itinéraire :
    central_lt = np.mean([itinerary[i]["latitude"] for i in range(0, len(itinerary))])
    central_lg = np.mean([itinerary[i]["longitude"] for i in range(0, len(itinerary))])

    ## Création de la carte  :
    m = flm.Map(location=[central_lt, central_lg], zoom_start=12, width=325, height=300)

    ## Ajout des pois de l'itinéraire  :
    for i in range(0, len(itinerary)):
        poi_lt = itinerary[i]["latitude"]
        poi_lg = itinerary[i]["longitude"]
        poi_sub_cat = itinerary[i]["sub_category"]
        icon = categories_data[poi_sub_cat]["icon"]
        color = categories_data[poi_sub_cat]["color"]
        popup_html = f"""
                        <div style="line-height: 1.3; margin-bottom: 15px; margin-top: 15px; margin-left: 10px;">
                            <i class="fa fa-{icon}" style="color: {color}; font-size: 18px;"></i> 
                            <b>{itinerary[i]['nom_du_poi']}</b><br>
                            <small>📍{itinerary[i]['adresse']}</small><br>
                            <small>☎️{itinerary[i]['contact_phone']}</small><br>
                            <small>📧{itinerary[i]['contact_mail']}</small><br>
                            <small>🌐<a href="{itinerary[i]['contact_website']}" target="_blank">{itinerary[i]['contact_website']}</a></small>
                            </div>
                    """
        

        flm.Marker(
            location=[poi_lt, poi_lg],
            popup= flm.Popup(popup_html, max_width=300),
            icon=flm.Icon(
                prefix="fa",
                icon=categories_data[poi_sub_cat]["icon"],
                color=categories_data[poi_sub_cat]["color"],
                icon_color="white",
            ),
        ).add_to(m)


    # Géométrie :
    geometry = itinerary_list[day]["geometry"]  # GeoJSON LineString
    coords = geometry["coordinates"]

    # Construire un FeatureCollection valide
    geojson_route = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {"day": day}
            }
        ]
    }

    layer = flm.FeatureGroup(name=f"Day {day}")
    ## Ajouter la route OSRM
    flm.GeoJson(
        geojson_route,
        style_function=lambda x, color=COLORS[day % len(COLORS)]: {
            "color": color,
            "weight": 4,
            "opacity": 0.9
        }
    ).add_to(layer)

    layer.add_to(m)


    # Récupération des catégories principales de l'itinéraire, de la durée et distance globales et du nombre de poi :
    sub_cat_itin = list(set([poi["sub_category"] for poi in itinerary]))
    day_total_distance = itinerary_list[day]["total_distance_km"]
    day_total_duration = itinerary_list[day]["total_duration_min"]
    poi_nbre = len(itinerary)


    
    
    #----------------------------------------------------------
    #      Affichage des résultats : Entête de la carte 
    #--------------------------------------------------------
    with st.expander(f"**Journée n°{day+1}**", expanded=False) :

        ### badges :
        badges_markdown = ""
        for cat in sub_cat_itin:
            emoji = categories_emoji[cat]
            color = categories_color[cat]
            badges_markdown += f":{color}-badge[{emoji} {cat}] "
        st.markdown(badges_markdown)

        ### synthèse : nombre pois, distance et durée globales
        col_info, col_button = st.columns([4, 1], vertical_alignment="center")

        with col_info:
            st.markdown(
                f":round_pushpin: {poi_nbre} POIs, :straight_ruler: {distance_print(day_total_distance)}, :hourglass: {time_print(day_total_duration)}"
            )

        with col_button:
            with st.container(horizontal_alignment="right", gap="medium") :
                
                if st.session_state.show_details.get(day, False):
                    if st.button("Masquer", key=f"hide_{day}"):
                        st.session_state.show_details[day] = False
                        st.rerun()
                else:
                    if st.button("Voir plus", key=f"more_{day}"):
                        st.session_state.show_details[day] = True
                        st.rerun()

        #--------------------------------------------
        #      Affichage de la carte 
        #--------------------------------------------

        if st.session_state.show_details.get(day, False):
            # 2 colonnes si détails activés
            col_carte, col_details = st.columns([1, 1])

            with col_carte:
                st_folium(m, width=500, height=400)

            with col_details:
                with st.container(height=400, border=False, vertical_alignment="distribute"):
                    st.markdown(
                        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">',
                        unsafe_allow_html=True,
                    )

                    for i in range(0, len(itinerary)):
                        if i == 0:
                            poi_cat = itinerary[i]["sub_category"]
                            icon = categories_data[poi_cat]["icon"]
                            color = categories_data[poi_cat]["color"]

                            with st.container(border=False) :
                                st.markdown(
                                    f"""
                                        <div style="background-color: #e6cba3; line-height: 1.3; margin-bottom: 5px; padding: 15px; border-radius: 10px;">
                                        <i class="fa fa-{icon}" style="color: {color}; font-size: 18px;"></i> 
                                        <b style="font-size: 18px;">{itinerary[i]['nom_du_poi']}</b><br>
                                        <small> 📍{itinerary[i]['adresse']}</small><br>
                                        <small> ☎️{itinerary[i]['contact_phone']}</small><br>
                                        <small> 📧{itinerary[i]['contact_mail']}</small><br>
                                        <small> 🌐<a href="{itinerary[i]['contact_website']}" target="_blank">{itinerary[i]['contact_website']}</a></small>
                                        </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            
                        else:
                            poi_cat = itinerary[i]["sub_category"]
                            icon = categories_data[poi_cat]["icon"]
                            color = categories_data[poi_cat]["color"]

                            d = itinerary[i]["distance_from_prev_km"]
                            t = itinerary[i]["duration_from_prev_min"]
                            
                            with st.container(border=False) :
                                st.markdown(
                                    f"""
                                        <div style="line-height: 1.3; margin-bottom: 10px; margin-left: 30px;">
                                        <small>{mobility_mode_icon} <b>Distance:</b>  {distance_print(d)}</small><br>
                                        <small>⏱️ <b>Durée</b>: {time_print(t)}</small><br>
                                        </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with st.container(border=False) :
                                st.markdown(
                                    f"""
                                        <div style="background-color: #e6cba3; line-height: 1.3; margin-bottom: 5px; padding: 15px; border-radius: 10px;">
                                        <i class="fa fa-{icon}" style="color: {color}; font-size: 18px;"></i> 
                                        <b style="font-size: 18px;">{itinerary[i]['nom_du_poi']}</b><br>
                                        <small> 📍{itinerary[i]['adresse']}</small><br>
                                        <small> ☎️{itinerary[i]['contact_phone']}</small><br>
                                        <small> 📧{itinerary[i]['contact_mail']}</small><br>
                                        <small> 🌐<a href="{itinerary[i]['contact_website']}" target="_blank">{itinerary[i]['contact_website']}</a></small>
                                        </div>
                                        """,
                                unsafe_allow_html=True,
                            )

        else:
            # Pleine largeur par défaut
            st_folium(m, width=1050, height=400)
            


# ## Ajouter un layer par jour
# for idx, day in enumerate(itinerary_list):
#     day_num = day["day"]
#     geometry = day["geometry"]  # GeoJSON LineString
#     coords = geometry["coordinates"]

#     # Construire un FeatureCollection valide
#     geojson_route = {
#         "type": "FeatureCollection",
#         "features": [
#             {
#                 "type": "Feature",
#                 "geometry": geometry,
#                 "properties": {"day": day_num}
#             }
#         ]
#     }

#    # Créer un FeatureGroup pour ce jour
#     layer = folium.FeatureGroup(name=f"Day {day_num}")

#     # Créer la carte
#     m = folium.Map(location=[day["pois"][0]["latitude"], day["pois"][0]["longitude"]], zoom_start=13)

#     # Ajouter la route OSRM
#     folium.GeoJson(
#         geojson_route,
#         style_function=lambda x, color=COLORS[idx % len(COLORS)]: {
#             "color": color,
#             "weight": 4,
#             "opacity": 0.9
#         }
#     ).add_to(layer)

#     # Ajouter les POIs du jour
#     for poi in day["pois"]:
#         folium.Marker(
#             location=[poi["latitude"], poi["longitude"]],
#             popup=poi["nom_du_poi"],
#             icon=folium.Icon(color="blue", icon="info-sign")
#         ).add_to(layer)
    
#     # Ajouter le layer à la carte
#     layer.add_to(m)

#     # Afficher la carte
#     st_folium(m, width=800, height=600)

