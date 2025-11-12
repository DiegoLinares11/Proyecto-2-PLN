import torch
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification, DebertaV2Tokenizer
import joblib

def load_distilbert():
    MODEL_NAME = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_labels = 3

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    model.load_state_dict(torch.load("./models/distilbert_trained.pt", map_location=device))            
    model.to(device)
    model.eval()

    return model

def load_deberta():
    model = AutoModelForSequenceClassification.from_pretrained("./models/deberta_model")
    tokenizer = DebertaV2Tokenizer.from_pretrained("./models/deberta_model")

    label_encoder = joblib.load("./models/deberta_model/label_encoder.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model
