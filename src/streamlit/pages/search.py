from datetime import timedelta

import folium as flm
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
from utils import fetch_main_categories, fetch_sub_categories, get_selected_pois, send_payload

import streamlit as st

# ------------------------------------
# import de données locales pour test
# ------------------------------------

df_localities = pd.read_csv("data/localities.csv")
localities = df_localities["locality"].unique().tolist()


# -----------------------------------------------------------

st.title("✨ TripMaNGo ✨", text_alignment="center")
st.header("Ne planifiez plus vos voyages, profitez-en !")
st.markdown(
    "💡 **TripMaNGo** est votre assistante de voyage. En quelques clics, créez un itinéraire personnalisé adapté à vos besoins de voyages"
)

st.divider()

st.subheader("🎯 Quels sont vos critères de recherche ? ")


# ---------------------------------------
## zone géographique
# ---------------------------------------

with st.expander("📍 **Quelle est votre destination ?**"):
    

    locality = st.multiselect(
        "1️⃣ **Je veux visiter cette destination**", localities, max_selections=1
    )
    radius = st.slider("2️⃣ **Dans un rayon de** (en km)", 1, st.session_state.max_radius)

    @st.cache_data
    def create_geo_zone_map(locality, radius):
        # récupérer latitude et longitude du centre de la localité
        latitude = df_localities.loc[
            df_localities["locality"] == locality[0], "center_latitude"
        ].values[0]
        longitude = df_localities.loc[
            df_localities["locality"] == locality[0], "center_longitude"
        ].values[0]

        # création d'une carte centrée sur les coordonnées du centre de la localité
        map = flm.Map(
            location=[latitude, longitude], zoom_start=10, width=600, height=400
        )

        # ajouter le point central :
        icon = flm.Icon(color="red", icon="")

        flm.Marker(
            location=[latitude, longitude],
            popup=locality[0],
            icon=icon,
        ).add_to(map)

        # ajouter la délimitaion de la zone géographique
        flm.Circle(
            location=[latitude, longitude],
            radius=radius * 1000,
            color="green",
            fill=True,
            fillColor="green",
            fillOpacity=0.2,
            weight=2,
        ).add_to(map)

        return map, longitude, latitude

    if locality and radius:
        map, longitude, latitude = create_geo_zone_map(locality, radius)
        st_folium(map, width=600, height=400)
        st.session_state.payload["longitude"] = longitude
        st.session_state.payload["latitude"] = latitude
        st.session_state.payload["commune"] = df_localities.loc[
            df_localities["locality"] == locality[0], "locality_name"
        ].values[0]
        st.session_state.payload["radius"] = radius

# ---------------------------------------
## durée du séjour
# ---------------------------------------

with st.expander("📅 **Quelle est la durée de votre séjour ?**"):
    options = [
        "saisir un nombre de jours",
        "saisir la date d'arrivée et la date de départ",
    ]
    choice = st.radio("3️⃣ **Je souhaite** :", horizontal=True, options=options)

    if choice == options[0]:
                    
        num_days = st.slider(
            "Nombre de jours", 
            1, 
            st.session_state.max_days,
            value=st.session_state.payload["days"],  
            key="num_days_slider"
        )
        # Mettre à jour immédiatement
        st.session_state.payload["days"] = num_days
        
    else:
        col1, col2 = st.columns(2)
        with col1:
            arrival_date = st.date_input(
                "Date d'arrivée", format="DD/MM/YYYY", min_value="today"
            )
        with col2:
            departure_date = st.date_input(
                "Date de départ",
                format="DD/MM/YYYY",
                min_value=arrival_date,
                max_value=(
                    arrival_date + timedelta(days=(st.session_state.max_days - 1))
                ),
            )

        if arrival_date and departure_date:
            num_days = (departure_date - arrival_date).days + 1
            st.text(f"Nombre de jours : {num_days}")
            if num_days < 0:
                st.error("❌ La date de départ doit être après la date d'arrivée")
            else:
                # Mettre à jour le payload
                st.session_state.payload["days"] = num_days

# ---------------------------------------
## Moyen de transport/mobilité
# ---------------------------------------

with st.expander("👣 **Comment souhaiteriez-vous vous déplacer ?**"):
    mobility_mean = st.selectbox(
        "5️⃣ **Moyen de mobilité/transport**", st.session_state.dict_mobility.keys()
    )

    transport_mode = st.session_state.dict_mobility[mobility_mean]
    
    if transport_mode:
        st.session_state.payload["transport_mode"] = transport_mode


# ---------------------------------------
## Préférences/activitées souhaitées
# ---------------------------------------

with st.expander("🎪 **Qu'est-ce qui vous ferait plaisir ?**"):
    # catégories principales :
    main_categories = fetch_main_categories()
    main_cat = st.multiselect(
        "5️⃣ **Catégorie(s) principale(s)**", main_categories, default=[]
    )

    if main_cat:
        sub_categories = fetch_sub_categories(main_cat)
        sub_cat = st.multiselect("6️⃣ **Sous-catégorie(s)**", sub_categories, default=[])

    if main_cat and sub_cat:
        st.session_state.payload["main_category"] = main_cat
        st.session_state.payload["sub_category"] = sub_cat

# ---------------------------------------------------------------------
## Validation des paramètres et lancement de la recherche
# ---------------------------------------------------------------------

if st.button("Proposer des itinéraires", type="primary"):
    payload = st.session_state.payload
    if (
        (payload["commune"] == "")
        or (payload["main_category"] == [])
        or (payload["sub_category"] == [])
        or (payload["days"] == 0)
        or (payload["latitude"] == 0)
        or (payload["longitude"] == 0)
        or (payload["transport_mode"] == "")
        or (payload["radius"] == 0)
    ):
        st.error("❌ Un ou plusieurs paramètres de recherche sont invalides")
    else:
        
        pois = get_selected_pois(payload)

        # Construction du payload pour /itinerary/compute
        itinerary_payload = {
            "pois": pois["pois"],
            "days": payload["days"] ,
            "transport_mode": payload["transport_mode"],
            "solver": "auto",
            "latitude": payload["latitude"],
            "longitude": payload["longitude"],
        }
        # Sauvegarde dans la session
        st.session_state.itinerary_payload = itinerary_payload

        # Supprimer le cache de la recherche d'itinéraire s'il existe: 
        if "itinerary_result" in st.session_state:
            del st.session_state.itinerary_result
        
        ## DEBUG
        #st.write(payload)
        #st.write(itinerary_payload)

        #st.subheader("Payload prêt pour /itinerary/compute")
        #st.json(itinerary_payload)

        # Redirection vers la page "Itinéraire"

        st.session_state.current_page = "results"
        st.rerun()

