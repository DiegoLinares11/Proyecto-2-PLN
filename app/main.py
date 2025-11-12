import torch
import streamlit as st
import load_models as lm

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

distilbert_model = lm.load_distilbert(device)
deberta_model = lm.load_deberta(device)

st.title("Argument Classifier")

model_choice = st.selectbox(
    "Select model to use:",
    ("DistilBERT", "DeBERTa")
)

argument = st.text_area(
    label="Argument", 
    placeholder="Write your argument...",
    key="argument_text"
)

if st.button("Classify argument", key="classify_btn"):
    if argument.strip() == "":
        st.warning("Please write an argument before classifying.")
    else:
        with st.spinner("Classifying..."):
            if model_choice == "DistilBERT":
                prediction = lm.predict_distilbert(argument, device, distilbert_model)
                st.session_state["last_prediction"] = f"Predicted class (DistilBERT): {prediction}"
            else:
                prediction = lm.predict_deberta(argument, device, deberta_model)
                st.session_state["last_prediction"] = f"Predicted label (DeBERTa): {prediction}"


if "last_prediction" in st.session_state: 
    st.text(f"Prediction made: {st.session_state['last_prediction']}")
