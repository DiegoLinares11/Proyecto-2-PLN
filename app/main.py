import streamlit as st

st.title("Hello")
st.markdown(
    """
    Playground for testing
    """
)

if st.button("Hello world"):
    st.balloons()

st.text_area(label="Argument", placeholder="Write your argument...")
