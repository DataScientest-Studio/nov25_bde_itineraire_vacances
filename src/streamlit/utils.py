import requests
import streamlit as st

## récupération des main_categories :
@st.cache_data
def fetch_main_categories() :
    main_cat_url = "http://localhost:8000/main_categories"
    try :
        response = requests.get(main_cat_url)
        response.raise_for_status() 
        data = response.json()
        return data['main_categories']
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")

## récupération des sub_categories :
@st.cache_data
def fetch_sub_categories(main_cat) :
    sub_cat_url = "http://localhost:8000/sub_categories"
    try :
        params= {"categories_list": main_cat}
        response = requests.post(sub_cat_url, json= params)
        response.raise_for_status() 
        data = response.json()
        return data['sub_categories']
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")