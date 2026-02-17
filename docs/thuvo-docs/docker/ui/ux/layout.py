# ======================================================================================================================
# PAGE CONFIG ---> docker/ui/ux/layout.py
# ======================================================================================================================
import streamlit as st

def set_page_config() -> None:
    st.set_page_config(page_title="Itinéraire Vacances • Prime", page_icon="🧭", layout="wide")


def sidebar_header():
    """
    Affiche l'en-tête du mode Prime dans la sidebar Streamlit.
    - Titre visuel fort (emoji + typographie)
    - Sous-titre explicatif pour guider l'utilisateur
    - Utilise du HTML inline pour un contrôle fin du rendu
    """
    st.markdown(
        """
        <div style="padding: 2px 0 6px 0;">
          <div style="font-size: 30px; font-weight: 800; line-height: 1.05;">🧭 Itinéraire de Vacances 🧳</div>
          <div style="color: #6b7280; margin-top: 6px; font-size: 13px;">
          </div>
        </div>
        """,
        unsafe_allow_html=True,)


def section_title(txt: str, sub: str = ""):
    """
    Affiche un titre de section cohérent dans la sidebar ou le contenu principal.
    Paramètres :
    - txt : titre principal (Markdown h4)
    - sub : sous-texte optionnel (description, aide utilisateur)
    Objectif :
    - Uniformiser le style des sections
    - Améliorer la lisibilité et la compréhension du flow
    """
    st.markdown(f"#### {txt}")
    if sub:
        st.markdown(
            f"<div style='color:#6b7280; font-size:13px; line-height:1.4'>{sub}</div>",
            unsafe_allow_html=True,)
        