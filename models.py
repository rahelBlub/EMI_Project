"""
models.py
==========
Die zwei Fusionsmodelle: Early Fusion und Cross-Attention

Was ist ein "Modell" hier?
--------------------------
Ein Modell nimmt die Embeddings (Zahlen-Listen) von Text und Bild
und sagt eine Klasse vorher (z.B. "positive", "negative").

Es hat intern lernbare Gewichte – diese passen sich beim Training an.
Die vortrainierten Encoder (SentenceTransformer, CLIP) bleiben eingefroren;
nur diese kleinen Köpfe hier werden trainiert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
# MODELL 1: EARLY FUSION
# ══════════════════════════════════════════════
class EarlyFusion(nn.Module):
    """
    Idee: Kleb Text- und Bild-Embedding einfach zusammen (384 + 512 = 896 Zahlen)
    und steck das durch ein neuronales Netz.

    Vorteil:  einfach, schnell, oft überraschend gut
    Nachteil: das Netz muss selbst herausfinden, welche Modalität wie wichtig ist

    Architektur:
        [h_t (384)] + [h_i (512)]
              ↓  concat → (896)
          LayerNorm    ← wichtig! Text- und Bild-Embeddings haben
              ↓          verschiedene "Größenordnungen", das gleichen wir aus
          Linear(896 → hidden)
              ↓
            ReLU
              ↓
           Dropout      ← verhindert Overfitting
              ↓
          Linear(hidden → n_classes)
              ↓
           Logits       ← rohe Scores, CrossEntropyLoss macht daraus Wahrscheinlichkeiten
    """

    def __init__(self, d_t: int = 384, d_i: int = 512,
                 n_classes: int = 5, hidden: int = 256, dropout: float = 0.3):
        super().__init__()

        # LayerNorm normalisiert jede Modalität separat bevor wir sie zusammenkleben
        # → verhindert dass eine Modalität die andere "übertönt"
        self.norm_t = nn.LayerNorm(d_t)
        self.norm_i = nn.LayerNorm(d_i)

        self.head = nn.Sequential(
            nn.Linear(d_t + d_i, hidden),  # 896 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), # 256 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),  # 128 → 5
        )

    def forward(self, h_t: torch.Tensor, h_i: torch.Tensor) -> torch.Tensor:
        """
        h_t: (batch_size, 384)  – Text-Embeddings
        h_i: (batch_size, 512)  – Bild-Embeddings
        gibt (batch_size, n_classes) zurück
        """
        # Normalisieren und zusammenkleben
        z = torch.cat([self.norm_t(h_t), self.norm_i(h_i)], dim=-1)  # (B, 896)
        return self.head(z)  # (B, n_classes)


# ══════════════════════════════════════════════
# MODELL 2: CROSS-ATTENTION FUSION
# ══════════════════════════════════════════════
class CrossAttentionFusion(nn.Module):
    """
    Idee: Text und Bild "schauen aufeinander" – das Modell lernt,
    welche Teile des Textes relevant für das Bild sind und umgekehrt.

    Warum ist das besser als Early Fusion?
    Beim Meme "Das ist doch nicht mein Ernst [Bild von Merkel]" soll das
    Modell lernen, dass der Text ironisch gemeint ist – das erkennt man nur,
    wenn man Text und Bild zusammen bewertet.

    Technisch: Wir projizieren beide Embeddings auf die gleiche Dimension (d_proj),
    dann lassen wir sie gegenseitig Attention auf den anderen berechnen.

    Architektur:
        h_t (384) → Projektion → (d_proj)  ←┐ gegenseitig
        h_i (512) → Projektion → (d_proj)  ←┘ Attention

        Text schaut auf Bild:  attn(Query=Text,  Key/Value=Bild)  → h_t_enriched
        Bild schaut auf Text:  attn(Query=Bild,  Key/Value=Text)  → h_i_enriched

        concat([h_t_enriched, h_i_enriched]) → Klassifikationskopf → Logits
    """

    def __init__(self, d_t: int = 384, d_i: int = 512,
                 d_proj: int = 256, n_heads: int = 4,
                 n_classes: int = 5, dropout: float = 0.3):
        super().__init__()

        # Beide Modalitäten auf gleiche Dimension projizieren (Voraussetzung für Attention)
        self.proj_t = nn.Linear(d_t, d_proj)  # Text:  384 → 256
        self.proj_i = nn.Linear(d_i, d_proj)  # Bild:  512 → 256

        # MultiheadAttention: "Welche Teile sind für wen relevant?"
        # n_heads=4 bedeutet: 4 parallele Attention-Mechanismen, die verschiedene
        # Aspekte der Beziehung zwischen Text und Bild lernen
        self.text_attends_to_image = nn.MultiheadAttention(
            embed_dim=d_proj, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.image_attends_to_text = nn.MultiheadAttention(
            embed_dim=d_proj, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # Nach Attention: normalisieren und stabilisieren
        self.norm_t = nn.LayerNorm(d_proj)
        self.norm_i = nn.LayerNorm(d_proj)

        # Klassifikationskopf: kombinierte Features → Klasse
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),  # 512 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, n_classes),    # 256 → 5
        )

    def forward(self, h_t: torch.Tensor, h_i: torch.Tensor) -> torch.Tensor:
        """
        h_t: (batch_size, 384)
        h_i: (batch_size, 512)
        gibt (batch_size, n_classes) zurück
        """
        # Beide auf (batch_size, 1, d_proj) bringen
        # Das "1" ist die Sequenzlänge – wir haben nur ein Token pro Modalität
        t = self.proj_t(h_t).unsqueeze(1)  # (B, 1, 256)
        i = self.proj_i(h_i).unsqueeze(1)  # (B, 1, 256)

        # Text schaut auf Bild: "Was am Bild ist relevant für den Text?"
        # Query = Text, Key/Value = Bild
        t_enriched, _ = self.text_attends_to_image(query=t, key=i, value=i)
        t_enriched = self.norm_t(t + t_enriched)  # Residual-Verbindung: "+t" damit Info nicht verloren geht

        # Bild schaut auf Text: "Was am Text ist relevant für das Bild?"
        # Query = Bild, Key/Value = Text
        i_enriched, _ = self.image_attends_to_text(query=i, key=t, value=t)
        i_enriched = self.norm_i(i + i_enriched)  # Residual-Verbindung

        # (B, 1, 256) → (B, 256) – Sequenzdimension wieder entfernen
        t_out = t_enriched.squeeze(1)
        i_out = i_enriched.squeeze(1)

        # Zusammenkleben und klassifizieren
        z = torch.cat([t_out, i_out], dim=-1)  # (B, 512)
        return self.head(z)  # (B, n_classes)
