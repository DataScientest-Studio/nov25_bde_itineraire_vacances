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

with st.container(key='body', height=520, border=False) :
    col1, col2 = st.columns(2)

    with col1 :
        with st.container(height=520, border=False):
            with st.expander("🔧**Perspectives techniques**", expanded=True):
                with st.container(height=390, border=False): 
                    st.markdown('''
                        - Intégrer de **nouvelles sources de donnnées**: <br>
                            -> transport public <br>
                            -> avis tripadvisor <br>
                            -> horaires d'ouvertures <br>
                            -> prix <br><br>
                        - **Optimiser le solveur GA** <br>
                            -> enrichir la fonction de sélection<br>
                            -> optimiser les hyperparamètres du modèle<br><br>
                        - Renforcer la **pipeline CI/CD**<br>
                        - Bêta **testing**<br>
                        ''', unsafe_allow_html=True)
    
    with col2 :
        with st.container(height=520, border=False): 
            with st.expander("💡**Perspectives Business**", expanded=True) :
                with st.container(height=390, border=False): 
                    st.markdown('''
                        - **UI/UX :** partage et suggestion d'itinéraire<br>
                                
                        - **Créer de l'engagement :** <br>
                            -> Mise en avant sponsorisée<br>
                            -> Intégration de billets<br>
                            -> Réductions partenaires<br>
                        - **Itinéraires collaboratifs :** <br>
                            -> avis sur les itinéraires<br>
                            -> badges créés pour fidéliser les utilisateurs <br>
                        - **Version premium** avec abonnement / **Version mobile** <br>  
                        - **Recherche sémantique** des POIs via des  thèmes de visite en utilisant un LLM<br>
                        
                        ''',
                        unsafe_allow_html=True)

    
with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/architecture.py")

    with col3:
        st.markdown('**8**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/remerciement.py")