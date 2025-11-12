import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification, DebertaV2Tokenizer, DistilBertTokenizerFast
import joblib
import pandas as pd

# SVM + TF-IDF
def load_svm():
    train_dataset = pd.read_csv("../data/clean_train.csv")

    X_train = train_dataset["discourse_text_clean"]
    y_train = train_dataset["label"]

    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        encoding='utf-8',
        ngram_range=(1,2),
        stop_words='english'
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    svm = LinearSVC(random_state=42)
    svm.fit(X_train_tfidf, y_train)

    return svm, tfidf

def predict_svm(argument, model, vectorizer):
    X_new = vectorizer.transform([argument])
    prediction = model.predict(X_new)

    return prediction[0]

# DistilBERT
def load_distilbert(device):
    MODEL_NAME = "distilbert-base-uncased"
    num_labels = 3

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    model.load_state_dict(torch.load("./models/distilbert_trained.pt", map_location=device))            
    model.to(device)
    model.eval()

    return model

def predict_distilbert(argument, device, model, model_name="distilbert-base-uncased"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    inputs = tokenizer(argument, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

# DeBERTa
def load_deberta(device):
    model = AutoModelForSequenceClassification.from_pretrained("./models/deberta_model")

    model.to(device)
    model.eval()

    return model

def predict_deberta(argument, device, model):
    tokenizer = DebertaV2Tokenizer.from_pretrained("./models/deberta_model")

    label_encoder = joblib.load("./models/deberta_model/label_encoder.pkl")

    inputs = tokenizer(argument, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label
