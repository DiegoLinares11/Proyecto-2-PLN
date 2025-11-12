import torch
import streamlit as st
import plotly.express as px
import pandas as pd
import load_models as lm

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

st.sidebar.title("Model Loading")
with st.sidebar.expander("Load models"):
    st.write("Loading models, please wait...")
    distilbert_model = lm.load_distilbert(device)
    deberta_model = lm.load_deberta(device)
    svm_model, tfidf_vectorizer = lm.load_svm()
st.sidebar.success("Models loaded succesfully")

st.title("Argument Classifier")

model_choice = st.selectbox(
    "Select model to use:",
    ("DistilBERT", "DeBERTa", "SVM + TF-IDF", "All Models")
)

argument = st.text_area(
    label="Argument", 
    placeholder="Write your argument...",
    key="argument_text",
    height=150
)

if st.button("Classify argument", key="classify_btn", use_container_width=True):
    if argument.strip() == "":
        st.warning("Please write an argument before classifying.")
    else:
        with st.spinner("Classifying..."):
            results = {}

            if model_choice in ["DistilBERT", "All Models"]:
                prediction = lm.predict_distilbert(argument, device, distilbert_model)
                results["DistilBERT"] = prediction

            if model_choice in ["DeBERTa", "All Models"]: 
                prediction = lm.predict_deberta(argument, device, deberta_model)
                results["DeBERTa"] = prediction

            if model_choice in ["SVM + TF-IDF", "All Models"]:
                prediction = lm.predict_svm(argument, svm_model, tfidf_vectorizer)
                results["SVM + TF-IDF"] = prediction

            st.subheader("Predictions")
            for model_name, pred in results.items():
                st.markdown(f"**{model_name}** -> `{pred}`")

            st.session_state["last_prediction"] = results

if "last_prediction" in st.session_state: 
    st.markdown("### Last Predictions")
    for model_name, pred in st.session_state["last_prediction"].items():
        st.text(f"{model_name}: {pred}")

show_performance = st.checkbox("Show model performance on test data")

if show_performance:
    st.subheader("Model Performance Comparison")

    with st.spinner("Evaluating models on test set... (this might take a minute)"):
        performance_data = lm.evaluate_models(device, distilbert_model, deberta_model, svm_model, tfidf_vectorizer)

    fig = px.bar(
        performance_data.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Performance Metrics (Test Data)",
        text_auto=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
