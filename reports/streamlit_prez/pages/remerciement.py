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


st.container(height=75, border=False)
st.divider()
with st.container(key='body', height=200, horizontal_alignment='center', vertical_alignment='center', border=False) :
    st.title("""Nous vous remercions de votre attention.""", text_alignment='center')
    st.header(" Des questions ?", text_alignment='center')
st.divider()
st.container(height=75, border=False)

with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/perspectives.py")

    with col3:
        st.markdown('**9**', text_alignment='center')
    
    with col5 :
        with st.container(horizontal_alignment="left"):
            if st.button(label = "", shortcut="Right", width=30 ) : 
                st.switch_page("pages/solver.py")
    