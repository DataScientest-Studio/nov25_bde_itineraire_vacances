import streamlit as st
import requests
import pandas as pd


st.title("Optimisateur d'itinéraire de vacances")
st.markdown("application pour planifier vos vacances en toute tranquilité")
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
    data = response.json()
    st.json(data)
