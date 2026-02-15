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


def display_ga():
    mermaid_code ="""
        graph TD
            Start["🎯 Début<br/>Nombre de générations: N"] 
            Init["1️⃣ Génération Population<br/>Population aléatoire<br/>d'itinéraires"]
            Select["2️⃣ Sélection<br/>Critères:<br/>- POI restaurant déjeuner<br/>- Durée max respectée"]            
            Check{"Génération<br/>inférieur à N?"}
            Cross["3️⃣ Crossover<br/>Combiner 2 itinéraires<br/>pour créer enfants"]
            Mutate{"4️⃣ Mutation<br/>Probabilité seuil?"}
            MutateYes["Modifier itinéraire<br/>aléatoirement"]
            NoMutate["Garder itinéraire<br/>inchangé"]
            NewGen["5️⃣ Recréer Génération<br/>Nouvelle population"]
            End["✅ Fin<br/>Meilleur itinéraire trouvé"]
                
            Start --> Init
            Init --> Select
            Select --> Check
            Check -->|Oui| Cross
            Cross --> Mutate
            Mutate -->|Oui| MutateYes
            Mutate -->|Non| NoMutate
            MutateYes --> NewGen
            NoMutate --> NewGen
            NewGen --> Select
            Check -->|Non| End
                
            style Start fill:#F5E6D3,stroke:#8B7355,stroke-width:2px,color:#333
            style Init fill:#E8D4C0,stroke:#8B7355,stroke-width:2px,color:#333
            style Select fill:#D4A574,stroke:#8B7355,stroke-width:2px,color:#fff
            style Check fill:#C4A57B,stroke:#8B7355,stroke-width:2px,color:#fff
            style Cross fill:#B8956A,stroke:#8B7355,stroke-width:2px,color:#fff
            style Mutate fill:#A08070,stroke:#8B7355,stroke-width:2px,color:#fff
            style MutateYes fill:#8B7355,stroke:#6B5344,stroke-width:2px,color:#fff
            style NoMutate fill:#8B7355,stroke:#6B5344,stroke-width:2px,color:#fff
            style NewGen fill:#7A6047,stroke:#6B5344,stroke-width:2px,color:#fff
            style End fill:#6B5344,stroke:#4A3C2A,stroke-width:2px,color:#fff  
        """
    st_mermaid(mermaid_code)


def display_nn2o():
    mermaid_code= '''
        graph TD
		    Start["<br/>🎯 Début: Matrices de durée <br/><br/>"]
		 
            Nn["<br/>1️⃣ Nearest Neighbor(NN) <br/><br/>"]
            
            Opt["<br/>2️⃣ 2-opt (amélioration locale) <br/><br/>"]  
                    
            End["<br/>3️⃣ Chemin final et coût optimisé <br/><br/>"]
                    
                        
            Start --> Nn
            Nn--> Opt
            Opt --> End
                    
                        
            style Start fill:#F5E6D3,stroke:#8B7355,stroke-width:2px,color:#333
            style Nn fill:#E8D4C0,stroke:#8B7355,stroke-width:2px,color:#333
            style Opt fill:#D4A574,stroke:#8B7355,stroke-width:2px,color:#fff
            style End fill:#C4A57B,stroke:#8B7355,stroke-width:2px,color:#fff
    
    '''
    st_mermaid(mermaid_code, width="500px", show_controls=False)


col1, col2 = st.columns(2)
with col1 :
    with st.expander('**⚡Nearest Neighbor 2‑Opt (NN2O)**', expanded=True):
        with st.container(height=450, border=False, gap=None) :
            display_nn2o()
    
with col2 :
    with st.expander('🧬**Genetic Algorithm (GA)**', expanded=True):
        with st.container(height=450, border=False, gap=None) :
            display_ga()

with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/remerciement.py")

    with col3:
        st.markdown('**10**', text_alignment='center')
    
    