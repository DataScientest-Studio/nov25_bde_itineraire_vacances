import streamlit as st

with st.sidebar:
    st.image("pages/media/tripmango_2_reduc.png", output_format='PNG')

pg = st.navigation([
    st.Page("pages/page_de_garde.py", title="1️⃣ Introduction"),
    st.Page("pages/contexte.py", title="2️⃣ Contexte & Enjeux"),
    st.Page("pages/data.py", title="3️⃣ Exploration des données"),
    st.Page("pages/etl.py", title="4️⃣ Collecte de données"),
    st.Page("pages/api.py", title="5️⃣ Exploitation des données"),
    st.Page("pages/solver.py", title="6️⃣ Algorithmes NN2O & GA"),
    st.Page("pages/demo.py", title="7️⃣ Démonstration"),
    st.Page("pages/architecture.py", title="8️⃣ Déploiement & Architecture"),
    st.Page("pages/perspectives.py", title="9️⃣ Perspectives"),
    st.Page("pages/remerciement.py", title="🙏 Remerciement")
])


pg.run()
