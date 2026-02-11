import streamlit as st

st.set_page_config(layout='wide')

with st.container(height=40, border=False) :
    col1, col2 = st.columns(2)

    with col1 :
        with st.container() :
            st.image("pages/media/liora-logo - Copy.svg", output_format="SVG")

    with col2 :
        with st.container(horizontal_alignment='right') :
            st.image("pages/media/tripmango_2_picto.png", output_format="PNG")

with st.container(key='body', height=520, border= False, horizontal_alignment='center') :
    col1, col2 = st.columns(2)
    with col1:
        with st.expander(label= '🔎**Exploration**', expanded=True) :
            st.markdown('''
                        -> **POI**<br>
                        -> Ratings et reviews<br>
                        -> Calcul de distance <br>
                        -> Météo<br>
                        ...
            ''', unsafe_allow_html=True)
    
        with st.expander(label= '⚠️**Contraintes**', expanded=True) :
            st.markdown('''
                         - **Qualité de la données** : manquantes, obsolètes, hétérogènes...<br>
                         - **Accès payant** à certaines données.<br>
                         - Distances et durées dépendent du **mode de transport**.<br>
                         - Contraintes **utilisateurs**<br>
                         - **Solveurs** avec des comportements différents<br>
                         - **Itinéraires** doivent être cohérents, lisibles et réalistes<br>
            ''', unsafe_allow_html=True)
    col2.space('small')
    with col2:
    
        with st.expander(label= '🧩**Approche agile avec MVP**', expanded=True) :
            st.markdown('''
                        **Source de données**
                        - **DataTourisme avec 3 flux** pour les régions :<br>
                            -> Île-de-France, <br>
                            -> Auvergne-Rhône-Alpes <br>
                            -> Bretagne.<br>
                        
                        **Cas d'usage nominal :**
                        - Proposer un itinéraire par jour sur la base de :<br>
                            -> une zone géographique<br>
                            -> préfèrences de l'utilisateur (activités touristiques) <br>
                        
                        **🎯 Notre approche : Construire un MVP évolutif et enrichissable**
            ''', unsafe_allow_html=True)

with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/contexte.py")

    with col3:
        st.markdown('**3**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/etl.py")