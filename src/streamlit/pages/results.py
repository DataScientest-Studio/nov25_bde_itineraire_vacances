import json

import folium as flm
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
from utils import (
    distance_print,
    fetch_main_categories,
    fetch_sub_categories,
    send_payload,
    time_print,
)

import streamlit as st

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
    # def update_mobility_mean():
    #     """Mise à jour du moyen de transport dans la payload"""
    #     st.session_state.payload["osrm_mode"] = st.session_state.dict_mobility[
    #         st.session_state.mobility_mean_widget
    #     ]

    # index = list(st.session_state.dict_mobility.values()).index(
    #     st.session_state.payload["osrm_mode"]
    # )
    # mobility_mean = st.selectbox(
    #     "Moyen de mobilité/transport",
    #     st.session_state.dict_mobility.keys(),
    #     index=index,
    #     key="mobility_mean_widget",
    #     on_change=update_mobility_mean,
    # )

    # filtre sur les catégories :
    main_categories = fetch_main_categories()

    def update_main_categories():
        """Mise à jour des catégories principales et réinitialisation des sous-catégories dans la payload"""

        st.session_state.payload["main_category"] = st.session_state.main_cat_widget
        st.session_state.payload["sub_category"] = []

    main_cat = st.multiselect(
        "Catégorie(s) principale(s)",
        main_categories,
        default=st.session_state.payload["main_category"],
        key="main_cat_widget",
        on_change=update_main_categories,
    )

    if main_cat:
        sub_categories = fetch_sub_categories(main_cat)
    else:
        sub_categories = []

    def update_sub_categories():
        """fonction pour mettre à jour les sub_categories dans la payload au
        au changement du champ correspondant"""
        st.session_state.payload["sub_categories"] = st.session_state.sub_cat_widget

    sub_cat = st.multiselect(
        "Sous-catégorie(s)",
        sub_categories,
        default=st.session_state.payload["sub_category"],
        key="sub_cat_widget",
        on_change=update_sub_categories,
    )

    if st.button("Mettre à jour"):
        payload = st.session_state.payload
        if (
            (payload["main_category"] == [])
            or (payload["sub_category"] == [])
            or (payload["days"] == 0)
            or (payload["osrm_mode"] == "")
            or (payload["radius"] == 0)
        ):
            st.error("❌ Un ou plusieurs paramètres de filtres sont invalides")
        else:
            st.write(st.session_state.payload)



itinerary = send_payload(st.session_state.itinerary_payload)
itinerary_list = itinerary["itinerary"]

# Affichage de la liste brute
st.write("Itinerary list :", itinerary_list)


for day in range(0, len(itinerary_list)):
    # récupération de l'itinéraire :
    itinerary = itinerary_list[day]["pois"]
    st.write(f"Jour {day + 1} :", itinerary)



# categories_data = {
#     "Nature & Paysages": {"icon": "tree", "color": "green"},
#     "Information Touristique": {"icon": "info-circle", "color": "blue"},
#     "Bien-être & Santé": {"icon": "spa", "color": "lightblue"},
#     "Famille & Enfants": {"icon": "child", "color": "pink"},
#     "Transports": {"icon": "car", "color": "gray"},
#     "Commodités": {"icon": "shopping-basket", "color": "orange"},
#     "Événements & Traditions": {"icon": "theater-masks", "color": "purple"},
#     "Commerce & Shopping": {"icon": "shopping-bag", "color": "lightred"},
#     "Gastronomie & Restauration": {"icon": "utensils", "color": "red"},
#     "Culture & Musées": {"icon": "landmark", "color": "darkpurple"},
#     "Santé & Urgences": {"icon": "hospital", "color": "darkred"},
#     "Hébergement": {"icon": "hotel", "color": "beige"},
#     "Sports & Loisirs": {"icon": "baseball", "color": "cadetblue"},
#     "Services & Mobilité": {"icon": "car-side", "color": "gray"},
#     "Loisirs & Clubs": {"icon": "mask", "color": "darkblue"},
#     "Camping & Plein Air": {"icon": "campground", "color": "darkgreen"},
#     "Patrimoine & Monuments": {"icon": "gopuram", "color": "orange"},
# }

# categories_emoji = {
#     "Nature & Paysages": "🌲",
#     "Information Touristique": "ℹ️",
#     "Bien-être & Santé": "🧘",
#     "Famille & Enfants": "👶",
#     "Transports": "🚗",
#     "Commodités": "🛒",
#     "Événements & Traditions": "🎭",
#     "Commerce & Shopping": "🛍️",
#     "Gastronomie & Restauration": "🍴",
#     "Culture & Musées": "🏛️",
#     "Santé & Urgences": "🏥",
#     "Hébergement": "🏨",
#     "Sports & Loisirs": "⚾",
#     "Services & Mobilité": "🚙",
#     "Loisirs & Clubs": "🎭",
#     "Camping & Plein Air": "⛺",
#     "Patrimoine & Monuments": "🏰",
# }

# categories_color = {
#     "Nature & Paysages": "green",
#     "Information Touristique": "blue",
#     "Bien-être & Santé": "blue",
#     "Famille & Enfants": "violet",
#     "Transports": "gray",
#     "Commodités": "orange",
#     "Événements & Traditions": "violet",
#     "Commerce & Shopping": "orange",
#     "Gastronomie & Restauration": "red",
#     "Culture & Musées": "violet",
#     "Santé & Urgences": "red",
#     "Hébergement": "gray",
#     "Sports & Loisirs": "blue",
#     "Services & Mobilité": "gray",
#     "Loisirs & Clubs": "violet",
#     "Camping & Plein Air": "green",
#     "Patrimoine & Monuments": "orange",
# }

# # ------------------------------------
# # Partie centrale avec les résultats
# # ------------------------------------

# st.header("🗺️ Nos propositions d'itinéraires")

# if "show_details" not in st.session_state:
#     st.session_state.show_details = {}

# # Récupération de l'icône du mode principale de mobilité :
# osrm_mode = st.session_state.payload["osrm_mode"]
# index = list(st.session_state.dict_mobility.values()).index(
#     st.session_state.payload["osrm_mode"]
# )
# mobility_mode_icon = list(st.session_state.dict_mobility.keys())[index][0]

# for day in range(0, len(results)):
#     # récupération de l'itinéraire :
#     itinerary = results[day]["pois"]

#     # création de la carte pour la journée :
#     ## détérmination du point central de la carte par rapport à l'itinéraire :
#     central_lt = np.mean([itinerary[i]["latitude"] for i in range(0, len(itinerary))])
#     central_lg = np.mean([itinerary[i]["longitude"] for i in range(0, len(itinerary))])

#     ## Création de la carte  :
#     m = flm.Map(location=[central_lt, central_lg], zoom_start=12, width=325, height=300)

#     ## Ajout des pois de l'itinéraire  :
#     for i in range(0, len(itinerary)):
#         poi_lt = itinerary[i]["latitude"]
#         poi_lg = itinerary[i]["longitude"]
#         poi_main_cat = itinerary[i]["main_category"]

#         flm.Marker(
#             location=[poi_lt, poi_lg],
#             popup=f"<b>{poi_main_cat}</b>",
#             icon=flm.Icon(
#                 prefix="fa",
#                 icon=categories_data[poi_main_cat]["icon"],
#                 color=categories_data[poi_main_cat]["color"],
#                 icon_color="white",
#             ),
#         ).add_to(m)

#     # Récupération des catégories principales de l'itinéraire, de la durée et distance globales et du nombre de poi :
#     main_cat_itin = list(set([poi["main_category"] for poi in itinerary]))
#     day_total_distance = itinerary[0]["day_total_distance"]
#     day_total_duration = itinerary[0]["day_total_duration"]
#     poi_nbre = len(itinerary)

#     ## Affichage des résulats :

#     ### Entête :
#     st.subheader(f"Journée n°{day+1}")

#     ### badges :
#     badges_markdown = ""
#     for cat in main_cat_itin:
#         emoji = categories_emoji[cat]
#         color = categories_color[cat]
#         badges_markdown += f":{color}-badge[{emoji} {cat}] "
#     st.markdown(badges_markdown)

#     ### synthèse : nombre pois, distance et durée globales
#     st.markdown(
#         f":round_pushpin: {poi_nbre} POIs, :straight_ruler: {distance_print(day_total_distance)}, :hourglass: {time_print(day_total_duration)}"
#     )

#     if st.session_state.show_details.get(day, False):
#         # 2 colonnes si détails activés
#         col_carte, col_details = st.columns([1, 1])

#         with col_carte:
#             st_folium(m, width=325, height=400)
#             if st.button("Masquer", key=f"hide_{day}"):
#                 st.session_state.show_details[day] = False
#                 st.rerun()

#         with col_details:
#             with st.container(height=400, border=True):
#                 st.markdown(
#                     '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">',
#                     unsafe_allow_html=True,
#                 )

#                 for i in range(0, len(itinerary)):
#                     if i == 0:
#                         poi_cat = itinerary[i]["main_category"]
#                         icon = categories_data[poi_cat]["icon"]
#                         color = categories_data[poi_cat]["color"]

#                         st.markdown(
#                             f"""
#                                     <div style="line-height: 1.3; margin-bottom: 15px; margin-left: 10px;">
#                                     <i class="fa fa-{icon}" style="color: {color}; font-size: 18px;"></i> <b>Point d'intérêt N°{i+1}</b><br>
#                                     <small>- Adresse: {itinerary[i]['adresse']}, {itinerary[i]['code_postal']}, {itinerary[i]['commune']}</small><br>
#                                     <small>- Contacts: <a href="{itinerary[i]['contacts_du_poi']}" target="_blank">{itinerary[i]['contacts_du_poi']}</a></small>
#                                     </div>
#                                     """,
#                             unsafe_allow_html=True,
#                         )
#                     else:
#                         poi_cat = itinerary[i]["main_category"]
#                         icon = categories_data[poi_cat]["icon"]
#                         color = categories_data[poi_cat]["color"]

#                         d = itinerary[i]["distance_from_prev"]
#                         t = itinerary[i]["duration_from_prev"]

#                         st.markdown(
#                             f"""
#                                         <div style="line-height: 1.3; margin-bottom: 15px; margin-top: 15px; margin-left: 20px;">
#                                         <small>{mobility_mode_icon} <b>Distance:</b>  {distance_print(d)}</small><br>
#                                         <small>⏱️ <b>Durée</b>: {time_print(t)}</small><br>
#                                         </div>
#                                         """,
#                             unsafe_allow_html=True,
#                         )

#                         st.markdown(
#                             f"""
#                                     <div style="line-height: 1.3; margin-bottom: 15px; margin-top: 15px; margin-left: 10px;">
#                                     <i class="fa fa-{icon}" style="color: {color}; font-size: 18px;"></i> <b>Point d'intérêt N°{i+1}</b><br>
#                                     <small>- Adresse: {itinerary[i]['adresse']}, {itinerary[i]['code_postal']}, {itinerary[i]['commune']}</small><br>
#                                     <small>- Contacts: <a href="{itinerary[i]['contacts_du_poi'][1:]}" target="_blank">{itinerary[i]['contacts_du_poi'][1:]}</a></small>
#                                     </div>
#                                     """,
#                             unsafe_allow_html=True,
#                         )

#     else:
#         # Pleine largeur par défaut
#         st_folium(m, width=700, height=400)
#         if st.button("Voir plus", key=f"more_{day}"):
#             st.session_state.show_details[day] = True
#             st.rerun()
