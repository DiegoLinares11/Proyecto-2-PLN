import torch
import streamlit as st
import load_models as lm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distilbert_model = lm.load_distilbert(device)
deberta_model = lm.load_deberta(device)

st.title("Argument Classifier")

argument = st.text_area(
    label="Argument", 
    placeholder="Write your argument...",
    key="argument_text"
)

if st.button("Classify argument", key="classify_btn"):
    st.session_state["last_prediction"] = argument

if "last_prediction" in st.session_state and argument != "":
    st.text(f"Prediction made: {st.session_state['last_prediction']}")
