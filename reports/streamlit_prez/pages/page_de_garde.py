import streamlit as st
from datetime import datetime

st.set_page_config(layout='wide')
today = datetime.today()
today = today.strftime("%d/%m/%Y")


with st.container(height=100, border=False) :
    col1, col2 = st.columns(2)

    with col1 :
        st.image("pages/media/liora-logo.svg", output_format="SVG")

    with col2 :
        st.subheader(f"**{today}**", text_alignment='right')

with st.container(vertical_alignment="center", horizontal_alignment="center", gap=None) : 
    st.image("pages/media/tripmango_2_reduc.png", output_format="PNG", )
    st.subheader("Ne planifiez plus vos vacances, profitez-en !", text_alignment="center")
  

st.divider()

with st.container(key='body', height=100, border=False) :
    col1, col2, col3 = st.columns(3, vertical_alignment="center")

    with col1 :
        st.markdown(f'''
                <div>
                <b>Cursus  : Data Engineer</b><br>
                <b>Projet  : Itinéraire de vacances</b><br>
                <b>Cohorte : DE - Bootcamp - novembre 2025</b>
                </div>
                ''',
                unsafe_allow_html=True)
    with col2 :
        st.markdown(f'''
                <div>
                <b>Equipe projet: </b><br>
                </div>
                ''',
                unsafe_allow_html=True, text_alignment='right' )
    with col3 :
        st.markdown(f'''
                <div>
                - Amadou ADJANOUHOUN<br>
                - Gérard PHAM<br>
                - Meriem GAZZAR
                </div>
                ''',
                unsafe_allow_html=True)

with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        st.markdown('**1**', text_alignment='center')

    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/contexte.py")