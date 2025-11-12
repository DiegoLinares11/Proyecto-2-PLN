import torch
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification, DebertaV2Tokenizer, DistilBertTokenizerFast
import joblib

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
