import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/search.py", title="Paramètres de recherche "),
        st.Page("pages/results.py", title="Consultation des résultats"),
    ]
)

# -------------------------------------------------
# Variables globales :
# -------------------------------------------------
## Nombre max de jours :
st.session_state.max_days = 30
st.session_state.max_radius = 30
st.session_state.dict_mobility = {
    "👟à pied": "walk",
    "🚗Voiture": "car",
    "🚴Vélo": "bike",
    "🚌Transport en commun": "public_transport",
}


# ------------------------------------------------------------
# Initialisation du payload de recherche d'itinéraire :
# ------------------------------------------------------------
if "payload" not in st.session_state:
    st.session_state.payload = {
        "commune": "",
        "main_category": [],
        "sub_category": [],
        "min_score": 0.15,
        "days": 0,
        "latitude": 0,
        "longitude": 0,
        "radius": 0,
        "osrm_mode": "",
        "solver": "auto",
    }

pg.run()
