import streamlit as st

st.set_page_config(layout='wide')

with st.container(height=40, border=False) :
    col1, col2 = st.columns(2)

    with col1 :
        with st.container() :
            st.image("pages/media/liora-logo - Copy.svg", output_format="SVG")

    with col2 :
        with st.container(horizontal_alignment='right') :
            st.image("pages/media/tripmango_reduc2.png", output_format="PNG")

with st.container(key='body', height=520, border=False) :
    col1, col2 = st.columns(2)

    with col1 :
        with st.container(height=520, border=False):
            with st.expander("🔧**Perspectives techniques**", expanded=True):
                with st.container(height=390, border=False): 
                    st.markdown('''
                        - Intégration de **nouvelles sources de donnnées**: <br>
                            -> transport public <br>
                            -> avis tripadvisor <br>
                            -> les horaires d'ouvertures <br>
                            -> Les prix <br> <br>
                        - **Optimisation du solveur GA** <br>
                            -> Enrichir la fonction de sélection<br>
                            -> optimiser des hyperparamètres du modèle<br><br>
                        - Renforcement de la **pipeline CI/CD**<br>
                        - Bêta **testing**<br>
                        ''', unsafe_allow_html=True)
    
    with col2 :
        with st.container(height=520, border=False): 
            with st.expander("💡**Perspectives Business**", expanded=True) :
                with st.container(height=390, border=False): 
                    st.markdown('''
                        - **Créer de l'engagement :** <br>
                            -> Mise en avant sponsorisée<br>
                            -> Intégration de billets<br>
                            -> Réductions partenaires<br>
                        - **Itinéraires collaboratifs :** <br>
                            -> avis sur les itinéraires<br>
                            -> badges créés pour fidéliser les utilisateurs <br>
                        - **Version premium** avec abonnement / **Version mobile** <br>  
                        
                        - **Recherche sémantique** des POIs :<br>
                            -> proposer des thèmes de visite en utilisant un LLM
                            <br>''',
                        unsafe_allow_html=True)

    
with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/architecture.py")

    with col3:
        st.markdown('**9**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/remerciement.py")