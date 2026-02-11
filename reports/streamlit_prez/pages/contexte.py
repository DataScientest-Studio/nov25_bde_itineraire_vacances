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
    
    col1, col2 = st.columns([0.75, 0.25], vertical_alignment="center")

    with col1 :
        with st.container(vertical_alignment='center', horizontal_alignment='center'):
            st.image("pages/media/TTDP-illustration.png", output_format="PNG", caption="**Tourist Trip Design Problem - TTDP -**")

    with col2 :
        
        with st.container(height=40, horizontal_alignment='center', vertical_alignment='center', border=False) :
            st.markdown(f'''**Expérience utilisateur**''')

        st.divider()

        with st.container(height=40, horizontal_alignment='center', vertical_alignment='center', border=False) :
            st.markdown(f'''**Challenge de la modélisation**''')

        st.divider()

        with st.container(height=40, horizontal_alignment='center', vertical_alignment='center', border=False) :
            st.markdown(f'''**Richesse des données**''')
            
        st.divider()

        with st.container(height=40, horizontal_alignment='center', vertical_alignment='center', border=False) :
                st.markdown(f'''**Diversité des outils**''')

with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/page_de_garde.py")

    with col3:
        st.markdown('**2**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/data.py")

        
