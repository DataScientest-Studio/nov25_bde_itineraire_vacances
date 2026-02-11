import streamlit as st
from streamlit_mermaid import st_mermaid

st.set_page_config(layout='wide')

with st.container(height=40, border=False) :
    col1, col2 = st.columns(2)

    with col1 :
        with st.container() :
            st.image("pages/media/liora-logo - Copy.svg", output_format="SVG")

    with col2 :
        with st.container(horizontal_alignment='right') :
            st.image("pages/media/tripmango_2_picto.png", output_format="PNG")


def display_graph(height=500):
    mermaid_code = """
        graph LR
        A["📊<br/>Source<br/>Datatourisme"]
        
        B1["Flux 1<br/><br/>POI Île-de-France"]
        B2["Flux 2<br/><br/>POI Bretagne"]
        B3["Flux 3<br/><br/>POI Auvergne-Rhône-Alpes"]
        
        C["<br/>📋Extraction<br/><br/> - Données nettoyées <br/><br/> - POI des 3 régions <br/><br/> >> 1 dataset <br/><br/>  "]
        
        D["<br/>⚙️Transformation<br/><br/>Nettoyage<br/><br/>Enrichissement : <br/><br/> - Catégorisation <br/> - Calcul H3 <br/> - Scoring <br/><br/> >> 1 dataset <br/><br/>"]
        
        E["<br/>🗃️ Initialisation BDD <br/><br/> Création des tables :<br/> - table POI<br/> - table Adresse<br/> - table catégories principales<br/> - table sous-catégories<br/><br/> Insertion des données :<br/> - catégories principales<br/> - sous-catégories<br/><br/>"]
        
        F["<br/>💾Chargement<br/><br/>Suppression des données :<br/> - table POI<br/> - table Adresse<br/><br/>  Insertion des données : <br/>- table POI<br/> - table Adresse<br/><br/>"]
        
        A --> B1
        A --> B2
        A --> B3
        
        B1 --> C
        B2 --> C
        B3 --> C
        
        C --> D
        D --> E
        E --> F
        
        style A fill:#F5E6D3,stroke:#8B7355,stroke-width:2px,color:#333
        style B1 fill:#D4A574,stroke:#8B7355,stroke-width:2px,color:#333
        style B2 fill:#D4A574,stroke:#8B7355,stroke-width:2px,color:#333
        style B3 fill:#D4A574,stroke:#8B7355,stroke-width:2px,color:#333
        style C fill:#C4A57B,stroke:#8B7355,stroke-width:2px,color:#333
        style D fill:#B8956A,stroke:#8B7355,stroke-width:2px,color:#333
        style E fill:#A08070,stroke:#8B7355,stroke-width:2px,color:#333,stroke-dasharray: 5 5
        style F fill:#8B7355,stroke:#6B5344,stroke-width:2px,color:#333
    """
    
    st_mermaid(mermaid_code)

with st.container(key='body', height=520, border=False, horizontal_alignment='center', vertical_alignment='center') :
#    st.image("pages/media/etl.svg", output_format="SVG")
    display_graph()



with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/data.py")

    with col3:
        st.markdown('**4**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/api.py")