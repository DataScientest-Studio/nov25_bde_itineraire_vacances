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
            st.image("pages/media/tripmango_reduc2.png", output_format="PNG")


def display_ga():
    st_mermaid("""
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
        """)


def display_nn2o():
    st_mermaid('''
        graph TD
            Start["🎯 Début<br/>Algorithme 2-Opt<br/>Solution initiale donnée"]
            Init["1️⃣ Initialisation<br/>Amélioration = Vrai<br/>Itération = 0"]
            Loop{"2️⃣ Amélioration<br/>trouvée?"}
            Select["3️⃣ Sélectionner 2 arêtes<br/>Arête i,i+1 et j,j+1<br/>où i < j"]
            Invert["4️⃣ Inverser segment<br/>Réordonner itinéraire<br/>entre positions i et j"]
            Calc["5️⃣ Calculer distance<br/>Comparer ancienne vs<br/>nouvelle distance"]
            Check{"6️⃣ Distance<br/>diminue?"}
            Apply["Appliquer inversion<br/>Mettre à jour solution"]
            NoApply["Annuler inversion<br/>Garder solution"]
            Check2{"7️⃣ Toutes les<br/>paires testées?"}
            LoopCheck{"8️⃣ Amélioration<br/>trouvée?"}
            End["✅ Fin<br/>Solution 2-Opt optimisée"]
            
            Start --> Init
            Init --> Loop
            Loop -->|Oui| Select
            Select --> Invert
            Invert --> Calc
            Calc --> Check
            Check -->|Oui| Apply
            Check -->|Non| NoApply
            Apply --> Check2
            NoApply --> Check2
            Check2 -->|Non| Loop
            Check2 -->|Oui| LoopCheck
            LoopCheck -->|Oui| Select
            LoopCheck -->|Non| End
            
            style Start fill:#F5E6D3,stroke:#8B7355,stroke-width:2px,color:#333
            style Init fill:#E8D4C0,stroke:#8B7355,stroke-width:2px,color:#333
            style Loop fill:#D4A574,stroke:#8B7355,stroke-width:2px,color:#fff
            style Select fill:#C4A57B,stroke:#8B7355,stroke-width:2px,color:#fff
            style Invert fill:#B8956A,stroke:#8B7355,stroke-width:2px,color:#fff
            style Calc fill:#A08070,stroke:#8B7355,stroke-width:2px,color:#fff
            style Check fill:#9B7B63,stroke:#8B7355,stroke-width:2px,color:#fff
            style Apply fill:#8B7355,stroke:#6B5344,stroke-width:2px,color:#fff
            style NoApply fill:#8B7355,stroke:#6B5344,stroke-width:2px,color:#fff
            style Check2 fill:#7A6047,stroke:#6B5344,stroke-width:2px,color:#fff
            style LoopCheck fill:#6B5344,stroke:#4A3C2A,stroke-width:2px,color:#fff
            style End fill:#4A3C2A,stroke:#2A1C0A,stroke-width:2px,color:#fff
    
    ''')


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
                st.switch_page("pages/api.py")

    with col3:
        st.markdown('**6**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/demo.py")
