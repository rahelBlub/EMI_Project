"""
train.py
=========
SCHRITT 2 VON 3: Training und Auswertung

Was passiert hier?
------------------
1. Wir laden die gespeicherten Embeddings (aus extract_features.py)
2. Wir teilen sie in Train/Val/Test auf (zufällig, da Memotion keinen Sprecher-Split braucht)
3. Wir trainieren beide Modelle (EarlyFusion und CrossAttention) je 3 Mal
4. Wir messen Macro-F1 und Accuracy, und erstellen eine Konfusionsmatrix

Voraussetzung:
--------------
  python extract_features.py   ← muss vorher laufen!
  pip install scikit-learn matplotlib seaborn
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

from models import EarlyFusion, CrossAttentionFusion


# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────
FEATURES_FILE = "C:/Users/rebek/Documents/EMI_Project/data/dataset/memotion_features.npz"
N_CLASSES     = 5
N_EPOCHS      = 40
BATCH_SIZE    = 64
LR            = 1e-3
SEEDS         = [42, 123, 999]   # 3 Seeds → Ergebnisse sind kein Zufall

LABEL_NAMES   = ["very_positive", "positive", "neutral", "negative", "very_negative"]


# ─────────────────────────────────────────────
# DATASET-KLASSE
# ─────────────────────────────────────────────
class MemotionDataset(Dataset):
    """
    Einfaches PyTorch-Dataset: gibt Text-Embedding, Bild-Embedding und Label zurück.
    PyTorch braucht diese Klasse, um Daten in Batches zu laden.
    """

    def __init__(self, h_t, h_i, labels):
        self.h_t    = torch.from_numpy(h_t).float()
        self.h_i    = torch.from_numpy(h_i).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.h_t[idx], self.h_i[idx], self.labels[idx]


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def set_seed(seed: int):
    """Alle Zufallsgeneratoren auf denselben Startwert setzen → reproduzierbar"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader) -> dict:
    """
    Lässt das Modell alle Daten im Loader einmal durchgehen und
    berechnet Macro-F1 und Accuracy.

    Macro-F1: Mittelwert des F1-Scores über alle Klassen
    (wichtig wenn Klassen ungleich groß sind – wie hier wahrscheinlich)
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():  # kein Gradient beim Auswerten nötig → spart Speicher
        for h_t, h_i, labels in loader:
            logits = model(h_t, h_i)
            preds  = logits.argmax(dim=-1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true":   y_true,
        "y_pred":   y_pred,
    }


def train_one_run(model, loader_train, loader_val, n_epochs: int, lr: float):
    """
    Trainiert ein Modell für n_epochs Epochen.
    Speichert den besten Zustand (nach Val-F1).

    Was ist eine Epoche? → Das Modell hat jeden Trainings-Datenpunkt einmal gesehen.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # Standard für Klassifikation

    best_f1    = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for h_t, h_i, labels in loader_train:
            optimizer.zero_grad()           # Gradienten vom letzten Schritt löschen
            logits = model(h_t, h_i)        # Vorhersage
            loss   = criterion(logits, labels)  # Wie falsch waren wir?
            loss.backward()                 # Gradienten berechnen
            optimizer.step()               # Gewichte anpassen
            total_loss += loss.item()

        # Nach jeder Epoche auswerten
        val_result = evaluate(model, loader_val)
        val_f1     = val_result["macro_f1"]

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Alle 10 Epochen Fortschritt ausgeben
        if (epoch + 1) % 10 == 0:
            print(f"    Epoche {epoch+1:2d}/{n_epochs}  "
                  f"Loss: {total_loss/len(loader_train):.4f}  "
                  f"Val-F1: {val_f1:.4f}")

    # Besten Zustand wiederherstellen
    model.load_state_dict(best_state)
    return model, best_f1


def plot_confusion_matrix(y_true, y_pred, model_name: str, filename: str):
    """Speichert eine Konfusionsmatrix als Bild – Pflicht laut Aufgabe."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Konfusionsmatrix – {model_name}")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Konfusionsmatrix gespeichert: {filename}")


def unimodal_baseline(h, labels, loader_val, name: str):
    """
    Trainiert ein einfaches lineares Modell auf nur einer Modalität.
    → Zeigt ob Fusion überhaupt etwas bringt (Pflicht laut Aufgabe).
    """
    d = h.shape[1]
    # Winziges Modell: direkt Linear ohne Zwischenschicht
    model = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_CLASSES))

    class UniDataset(Dataset):
        def __init__(self, h, labels):
            self.h = torch.from_numpy(h).float()
            self.y = torch.from_numpy(labels).long()
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.h[i], self.y[i]

    # Nur der Vollständigkeit halber – trainieren wir kurz
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    train_idx, _ = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
    loader = DataLoader(UniDataset(h[train_idx], labels[train_idx]), batch_size=64, shuffle=True)

    for _ in range(30):
        model.train()
        for x, y in loader:
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()

    # Auswerten auf Validation
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for h_t, h_i, y in loader_val:
            # Nutze je nach Modalität h_t oder h_i
            x = h_t if name == "Text" else h_i
            preds = model(x).argmax(dim=-1)
            y_true.extend(y.tolist()); y_pred.extend(preds.tolist())

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    print(f"  Unimodale Baseline ({name}): F1={f1:.4f}, Acc={acc:.4f}")
    return f1, acc


# ─────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Embeddings laden
    print("Lade Features...")
    data   = np.load(FEATURES_FILE)
    h_t    = data["h_t"]    # (N, 384)
    h_i    = data["h_i"]    # (N, 512)
    labels = data["labels"] # (N,)
    print(f"  Datensatz: {len(labels)} Beispiele")

    # 2. Train/Val/Test Split (80/10/10)
    # Wir machen das zufällig – Memotion hat keine Sprecher wie RAVDESS
    idx = np.arange(len(labels))
    idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    idx_val,  idx_test  = train_test_split(idx_temp, test_size=0.5, random_state=42,
                                           stratify=labels[idx_temp])

    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # 3. DataLoader erstellen
    loader_train = DataLoader(MemotionDataset(h_t[idx_train], h_i[idx_train], labels[idx_train]),
                              batch_size=BATCH_SIZE, shuffle=True)
    loader_val   = DataLoader(MemotionDataset(h_t[idx_val],   h_i[idx_val],   labels[idx_val]),
                              batch_size=BATCH_SIZE, shuffle=False)
    loader_test  = DataLoader(MemotionDataset(h_t[idx_test],  h_i[idx_test],  labels[idx_test]),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 4. Unimodale Baselines (zum Vergleich)
    print("\n── Unimodale Baselines ──")
    text_f1,  text_acc  = unimodal_baseline(h_t, labels[idx_train], loader_val, "Text")
    image_f1, image_acc = unimodal_baseline(h_i, labels[idx_train], loader_val, "Bild")

    # 5. Beide Fusionsmodelle mit je 3 Seeds trainieren
    results = {}  # { "EarlyFusion": [(f1_s1, acc_s1), ...], ... }

    model_configs = [
        ("EarlyFusion",        lambda: EarlyFusion(n_classes=N_CLASSES)),
        ("CrossAttentionFusion", lambda: CrossAttentionFusion(n_classes=N_CLASSES)),
    ]

    for model_name, model_factory in model_configs:
        print(f"\n── {model_name} ──")
        seed_results = []

        for seed in SEEDS:
            print(f"  Seed {seed}:")
            set_seed(seed)

            model = model_factory()
            model, best_val_f1 = train_one_run(
                model, loader_train, loader_val, N_EPOCHS, LR
            )

            # Test-Auswertung mit dem besten Modell
            test_result = evaluate(model, loader_test)
            seed_results.append((test_result["macro_f1"], test_result["accuracy"]))
            print(f"    → Test F1: {test_result['macro_f1']:.4f}, "
                  f"Acc: {test_result['accuracy']:.4f}")

        results[model_name] = seed_results

    # 6. Ergebnistabelle ausgeben
    print("\n" + "="*60)
    print("ERGEBNISTABELLE (Mittelwert ± Standardabweichung über 3 Seeds)")
    print("="*60)
    print(f"{'Methode':<25} {'Macro-F1':>12} {'Accuracy':>12}")
    print("-"*60)

    # Baselines
    print(f"{'Baseline Text only':<25} {text_f1:>12.4f} {text_acc:>12.4f}")
    print(f"{'Baseline Image only':<25} {image_f1:>12.4f} {image_acc:>12.4f}")

    best_model_name = None
    best_mean_f1    = 0.0

    for model_name, seed_results in results.items():
        f1s  = [r[0] for r in seed_results]
        accs = [r[1] for r in seed_results]
        mean_f1  = np.mean(f1s);  std_f1  = np.std(f1s)
        mean_acc = np.mean(accs); std_acc = np.std(accs)
        print(f"{model_name:<25} {mean_f1:.4f}±{std_f1:.4f}  "
              f"{mean_acc:.4f}±{std_acc:.4f}")
        if best_model_name is None:
            best_mean_f1    = mean_f1
            best_model_name = model_name

    print("="*60)

    # 7. Konfusionsmatrix für das beste Modell (letzter Seed reicht als Beispiel)
    print(f"\nErstelle Konfusionsmatrix für {best_model_name}...")
    set_seed(SEEDS[0])
    best_factory = dict(model_configs)[best_model_name]
    best_model   = best_factory()
    best_model, _ = train_one_run(best_model, loader_train, loader_val, N_EPOCHS, LR)
    final_result  = evaluate(best_model, loader_test)

    plot_confusion_matrix(
        final_result["y_true"], final_result["y_pred"],
        model_name=best_model_name,
        filename=f"confusion_matrix_{best_model_name}.png"
    )

    # 8. Ablation: Was passiert wenn wir eine Modalität auf 0 setzen?
    print("\n── Modalitäts-Ablation (nur für bestes Modell) ──")
    best_model.eval()

    for ablate in ["text", "image"]:
        y_true, y_pred = [], []
        with torch.no_grad():
            for h_t_b, h_i_b, y_b in loader_test:
                if ablate == "text":
                    h_t_b = torch.zeros_like(h_t_b)  # Text auf 0 setzen
                else:
                    h_i_b = torch.zeros_like(h_i_b)  # Bild auf 0 setzen
                preds = best_model(h_t_b, h_i_b).argmax(dim=-1)
                y_true.extend(y_b.tolist())
                y_pred.extend(preds.tolist())

        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  Ohne {ablate:6s}: F1 = {f1:.4f}  "
              f"(Rückgang: {best_mean_f1 - f1:+.4f})")

    print("\n✓ Fertig! Alle Ergebnisse ausgegeben.")
    print("  Tipp: Kopiere die Tabelle in deine README.md")
