import streamlit as st

with st.container(key='Top', border=True) :
    st.header("titre", text_alignment="center", )

with st.container(key='body', height=400) :
    st.markdown('corps de la slide')

with st.container(key='bottom') :
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        with st.container(horizontal_alignment="right"):
            if st.button(label = "", shortcut="Left", width=30) :
                st.switch_page("pages/remerciement.py")

    with col3:
        st.markdown('**10**', text_alignment='center')
    

    