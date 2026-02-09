import streamlit as st

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
}

# ------------------------------------------------------------
# Initialisation du payload de recherche d'itinéraire :
# ------------------------------------------------------------
if "payload" not in st.session_state:
    st.session_state.payload = {
        "commune": "",
        "latitude": 0,
        "longitude": 0,
        "main_category": [],
        "sub_category": [],
        "radius": 0,
        "days": 0,
        "transport_mode": "walk", 
        "solver": "auto"      
    }

if "itinerary_payload" not in st.session_state:
    st.session_state.itinerary_payload = {
        "pois": [],
        "days": 0,
        "transport_mode": "walk",
        "solver": "auto",
        "latitude": 0,
        "longitude": 0,        
    }

# Initialiser la page par défaut
if "current_page" not in st.session_state:
    st.session_state.current_page = "search"


# définir le layout selon la page
if st.session_state.current_page == "results":
    st.set_page_config(page_title="TripMango", layout="wide")
else:
    st.set_page_config(page_title="TripMango", layout="centered")


# Cacher le menu de navigation automatique
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)



with st.sidebar:
    with st.container(horizontal_alignment="center"):
        st.image("pages/media/tripmango.png", )
    
    st.divider()

    current = st.session_state.get("current_page", "search")
    
    # Bouton Recherche
    if st.button(
        "🔍 Recherche d'itinéraires", 
        use_container_width=True,
        type="primary" if st.session_state.current_page == "search" else "secondary"
    ):
        st.session_state.current_page = "search"
        st.rerun()
    
    # Bouton Résultats
    if st.button(
        "🗺️ Résultats", 
        use_container_width=True,
        type="primary" if st.session_state.current_page == "results" else "secondary"
    ):
        st.session_state.current_page = "results"
        st.rerun()
    
    st.divider()


#  Page à aggicher par défaut :
if "page" not in st.session_state:
    st.session_state.page = "search"

# -------------------------------------------------
# Afficher la page correspondante
# -------------------------------------------------
if st.session_state.current_page == "search":
    exec(open("pages/search.py").read())
elif st.session_state.current_page == "results":
    exec(open("pages/results.py").read())
    


