# ============================================================
# DistilBERT - Carga de pesos (Modelo 2) y generación de figuras
# ============================================================

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import matplotlib
matplotlib.use("Agg")  # backend no interactivo (evita errores en servidores/CI)
import matplotlib.pyplot as plt

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)

# -------------------------
# Rutas
# -------------------------
ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(ROOT, "distilbert_trained.pt")
VAL_CSV    = os.path.join(ROOT, "data", "clean_val_split.csv")
OUT_DIR    = os.path.join(ROOT, "outputs", "model2_results")

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No encuentro el modelo en: {MODEL_PATH}")
if not os.path.isfile(VAL_CSV):
    raise FileNotFoundError(f"No encuentro el CSV de validación en: {VAL_CSV}")

print(f"[INFO] MODEL_PATH = {MODEL_PATH}")
print(f"[INFO] VAL_CSV    = {VAL_CSV}")
print(f"[INFO] OUT_DIR    = {OUT_DIR}")

# -------------------------
# Configuración
# -------------------------
SEED         = 42
MAX_LENGTH   = 128
BATCH_SIZE   = 16
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LABEL2NAME   = {0: "Ineffective", 1: "Adequate", 2: "Effective"}
NAME2LABEL   = {v: k for k, v in LABEL2NAME.items()}
NUM_LABELS   = 3

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
print(f"[INFO] DEVICE = {DEVICE}")

# -------------------------
# Cargar validación
# -------------------------
df = pd.read_csv(VAL_CSV)

TEXT_COL_CANDIDATES  = ["discourse_text", "discourse_text_clean", "text", "fragment_text"]
LABEL_COL_CANDIDATES = ["label", "discourse_effectiveness_label", "y"]

text_col  = next((c for c in TEXT_COL_CANDIDATES  if c in df.columns), None)
label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)

if text_col is None:
    raise ValueError(f"No encuentro columna de texto. Esperaba alguna de: {TEXT_COL_CANDIDATES}")
if label_col is None:
    if "discourse_effectiveness" in df.columns:
        df["label"] = df["discourse_effectiveness"].map(NAME2LABEL)
        label_col = "label"
    else:
        raise ValueError(f"No encuentro columna de label. Esperaba alguna de: {LABEL_COL_CANDIDATES} o 'discourse_effectiveness'.")

# Mapear etiquetas en texto a numéricas si aplica
if df[label_col].dtype == object:
    if set(df[label_col].unique()) <= set(NAME2LABEL.keys()):
        df[label_col] = df[label_col].map(NAME2LABEL)
    else:
        raise ValueError("La columna de etiqueta es string pero no coincide con {Ineffective, Adequate, Effective}.")

df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

# -------------------------
# Tokenizador y modelo
# -------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS,
).to(DEVICE)

# -------------------------
# Carga robusta de pesos .pt
# -------------------------
def robust_load(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        try:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
        except Exception:
            if "model_state_dict" in ckpt:
                missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
                missing, unexpected = [], []
    else:
        # Menos común: objeto de modelo guardado tal cual
        model = ckpt
        missing, unexpected = [], []
    if missing or unexpected:
        print("[AVISO] Claves faltantes:", missing)
        print("[AVISO] Claves inesperadas:", unexpected)
    return model

model = robust_load(model, MODEL_PATH)
model.eval()

# -------------------------
# Dataset / DataLoader
# -------------------------
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx], int(self.labels[idx])

def collate_fn(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return {k: v for k, v in enc.items()}, labels

val_ds = TextDataset(df[text_col].tolist(), df[label_col].tolist())
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# -------------------------
# Evaluación
# -------------------------
all_preds, all_probs, all_labels = [], [], []
softmax = torch.nn.Softmax(dim=-1)

with torch.no_grad():
    for enc, labels in tqdm(val_loader, desc="Evaluando Modelo 2 (DistilBERT)"):
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        labels = labels.to(DEVICE)
        outputs = model(**enc)
        probs = softmax(outputs.logits).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

all_probs  = np.concatenate(all_probs, axis=0)
all_preds  = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

acc      = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")
rep = classification_report(
    all_labels, all_preds,
    target_names=[LABEL2NAME[i] for i in range(NUM_LABELS)],
    digits=4
)

print(f"[Modelo 2] Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f}")
print(rep)

with open(os.path.join(OUT_DIR, "metrics_model2.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\nF1-macro: {f1_macro:.4f}\n\n{rep}")
with open(os.path.join(OUT_DIR, "metrics_model2.json"), "w", encoding="utf-8") as f:
    json.dump({"accuracy": acc, "f1_macro": f1_macro}, f, indent=2)

# -------------------------
# Figuras
# -------------------------
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(5,4))
plt.imshow(cm_norm, interpolation='nearest')
plt.title("Matriz de confusión (normalizada) - Modelo 2")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.xticks([0,1,2], [LABEL2NAME[i] for i in range(NUM_LABELS)], rotation=45)
plt.yticks([0,1,2], [LABEL2NAME[i] for i in range(NUM_LABELS)])
for i in range(NUM_LABELS):
    for j in range(NUM_LABELS):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_model2.png"), dpi=200)
plt.close()

prec, rec, f1c, sup = precision_recall_fscore_support(
    all_labels, all_preds, labels=[0,1,2], zero_division=0
)

plt.figure(figsize=(6,4))
x = np.arange(NUM_LABELS)
plt.bar(x, f1c)
plt.xticks(x, [LABEL2NAME[i] for i in range(NUM_LABELS)], rotation=45)
plt.ylabel("F1 (por clase)")
plt.title("F1 por clase - Modelo 2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f1_per_class_model2.png"), dpi=200)
plt.close()

plt.figure(figsize=(6,4))
metrics_names = ["Accuracy", "F1-macro"]
metrics_vals  = [acc, f1_macro]
x = np.arange(len(metrics_names))
plt.bar(x, metrics_vals)
plt.xticks(x, metrics_names)
plt.ylim(0.0, 1.0)
plt.title("Desempeño global - Modelo 2 (DistilBERT)")
for i, v in enumerate(metrics_vals):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_metrics_model2.png"), dpi=200)
plt.close()

print(f"[OK] Figuras y métricas guardadas en: {OUT_DIR}")
