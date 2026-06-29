"""
extract_features.py
====================
SCHRITT 1 VON 3: Feature-Extraktion

Was passiert hier?
------------------
Wir nehmen jedes Meme aus dem Datensatz und lassen zwei vortrainierte
KI-Modelle drüberlaufen:
  - SentenceTransformer: liest den Text des Memes → gibt 384 Zahlen zurück
  - CLIP:                schaut das Bild an     → gibt 512 Zahlen zurück

Diese Zahlen-Listen ("Embeddings") speichern wir einmalig ab.
Danach brauchen wir die rohen Bilder/Texte nicht mehr – das spart beim
Training sehr viel Zeit.

Voraussetzung:
--------------
  pip install sentence-transformers transformers torch pillow pandas tqdm numpy

Dateistruktur erwartet:
-----------------------
  memotion_dataset_7k/
      labels.csv          ← deine bereits funktionierende CSV
      images/             ← Ordner mit den Meme-Bildern
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import dtype, float64, ndarray
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm  # Fortschrittsbalken
import clip

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

from src.data_loader import ExtractFeaturesKaggle
# ─────────────────────────────────────────────
# KONFIGURATION – hier kannst du Pfade anpassen
# ─────────────────────────────────────────────

extract_features = ExtractFeaturesKaggle()

DATASET_DIR  = "data/dataset/memotion_dataset_7k"          # Ordner mit dem Datensatz
LABELS_CSV   = os.path.join(DATASET_DIR, "labels.csv")
#IMAGE_DIR    = os.path.join(DATASET_DIR, "images")
OUTPUT_FILE  = "data/dataset/memotion_features.npz"        # hier landen die Embeddings
IMG_DIR = extract_features.get_images_path()

# extract_features.load_and_save_dataset()

dataset_labels = extract_features.load_dataset_from_dir()


# ─────────────────────────────────────────────
# MODELLE LADEN (einmalig, dann eingefroren)
# ─────────────────────────────────────────────
print("Lade Modelle... (nur beim ersten Mal etwas langsamer)")

# Text-Modell: versteht Sätze und gibt 384 Zahlen zurück
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Bild-Modell: versteht Bilder und gibt 512 Zahlen zurück
clip_model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_proc   = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Modelle geladen ✓")


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def get_text_embedding(text: str) -> np.ndarray:
    """
    Wandelt einen Text-String in einen 384-dimensionalen Vektor um.
    Der Vektor beschreibt die "Bedeutung" des Textes als Zahlen.
    """
    # encode() gibt direkt ein numpy-Array zurück
    return text_model.encode(str(text), show_progress_bar=False)  # (384,)


def get_image_embedding(image_path: str) -> Tensor | Module | ndarray[tuple[int], dtype[float64]]:
    """
    Wandelt ein Bild in einen 512-dimensionalen Vektor um.
    Gibt bei Fehler einen Nullvektor zurück, damit das Script nicht abbricht.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = clip_proc(images=img, return_tensors="pt")

        with torch.no_grad():  # kein Gradient nötig – wir trainieren CLIP nicht
            features = clip_model.get_image_features(**inputs)  # (1, 512)

        #return clip_model.image_embeds
        return features.squeeze(0).numpy()  # (512,)

    except Exception as e:
        print(f"  ⚠ Fehler bei {image_path}: {e} → Nullvektor wird verwendet")
        return np.zeros(512)


# ─────────────────────────────────────────────
# LABELS LADEN
# ─────────────────────────────────────────────
print(f"\nLese Labels aus {LABELS_CSV} ...")
#df = pd.read_csv(LABELS_CSV)
df = dataset_labels

# Zeige die ersten Zeilen, damit du siehst ob die Spalten stimmen
print(df.head(3))
print(f"Spalten: {list(df.columns)}")

# !! PASSE DIESE SPALTENNAMEN AN deine labels.csv an !!
# Typische Spalten in Memotion: 'image_name', 'text_corrected', 'humour', 'overall_sentiment'
IMAGE_COL  = "image_name"        # Spalte mit dem Dateinamen des Bildes
TEXT_COL   = "text_corrected"    # Spalte mit dem Text des Memes
LABEL_COL  = "overall_sentiment" # Spalte mit dem Label (was wir vorhersagen wollen)

# Memotion hat manchmal viele Label-Spalten. Wir nehmen 'overall_sentiment'
# Mögliche Werte: very_positive, positive, neutral, negative, very_negative
# Diese wandeln wir in Zahlen um:
label_map = {
    "very_positive": 0,
    "positive":      1,
    "neutral":       2,
    "negative":      3,
    "very_negative": 4,
}

# Zeilen mit fehlenden Labels oder Bildern rausfiltern
df = df.dropna(subset=[IMAGE_COL, TEXT_COL, LABEL_COL])
df = df[df[LABEL_COL].isin(label_map)]  # nur bekannte Labels behalten
df["label_int"] = df[LABEL_COL].map(label_map)

print(f"\n{len(df)} Beispiele nach Filterung")
print(f"Label-Verteilung:\n{df[LABEL_COL].value_counts()}\n")


# ─────────────────────────────────────────────
# EMBEDDINGS EXTRAHIEREN (der eigentliche Job)
# ─────────────────────────────────────────────
print("Extrahiere Features... (das dauert ein paar Minuten)")

all_text_embs  = []
all_image_embs = []
all_labels     = []
all_indices    = []  # merken welche Zeilen erfolgreich waren
#df = df.head(4000)  # ← nur zum Testen, danach wieder entfernen
for idx, row in tqdm(df.iterrows(), total=len(df)):

    # Text-Embedding
    h_t = get_text_embedding(row[TEXT_COL])

    # Bild-Embedding
    img_path = os.path.join(IMG_DIR, row[IMAGE_COL])
    #h_i = get_image_embedding(img_path)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    #h_i = model.encode_image(image).detach().cpu().numpy()
    h_i = model.encode_image(image).detach().cpu().numpy().squeeze(0)

    all_text_embs.append(h_t)
    all_image_embs.append(h_i)
    all_labels.append(row["label_int"])
    all_indices.append(idx)

print("\nSpeichere Features...")

# Alles in eine einzige Datei – danach brauchen wir die Bilder/Texte nicht mehr
np.savez(
    OUTPUT_FILE,
    h_t    = np.stack(all_text_embs),   # (N, 384) – Text-Embeddings
    h_i    = np.stack(all_image_embs),  # (N, 512) – Bild-Embeddings
    labels = np.array(all_labels),       # (N,)     – Labels als Zahlen
)

print(f"\n✓ Fertig! Gespeichert in '{OUTPUT_FILE}'")
print(f"  Text-Embeddings:  {np.stack(all_text_embs).shape}")
print(f"  Bild-Embeddings:  {np.stack(all_image_embs).shape}")
print(f"  Labels:           {np.array(all_labels).shape}")
print("\nWeiter mit: python train.py")
