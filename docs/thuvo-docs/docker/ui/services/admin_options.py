# ======================================================================================================================
# NORMALISATION ADMIN : region / department / postal_code ---> docker/ui/services/admin_options.py
# ======================================================================================================================

import streamlit as st
from data.normalization_admin import list_regions, list_departments


@st.cache_data(ttl=3600)
def get_regions(df_):
    return list_regions(df_)


@st.cache_data(ttl=3600)
def get_departments(df_, region: str | None):
    return list_departments(df_, region)
