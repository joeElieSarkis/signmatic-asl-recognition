import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced_normalized\data")
MODEL_PATH = BASE_PATH.parent / "models" / "best_face_4w_ctc_char.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 411
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CTCTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL)
        )

        self.temporal_conv = nn.Conv1d(
            D_MODEL, D_MODEL, kernel_size=3, stride=2, padding=1
        )

        self.pos = PositionalEncoding(D_MODEL, 30)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_LAYERS)
        self.fc = nn.Linear(D_MODEL, vocab_size)

    def forward(self, x):
        x = self.input_proj(x)

        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)

        x = self.pos(x)
        x = self.encoder(x)
        return self.fc(x)


# =========================
# LOAD TRAIN LABELS → BUILD VOCAB
# =========================
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [list(line.strip()) for line in f]


train_labels = load_labels(BASE_PATH / "y_train_face_4w_balanced_norm.txt")

chars = set()
for seq in train_labels:
    chars.update(seq)

vocab = {c: i for i, c in enumerate(sorted(chars))}
vocab["<blank>"] = len(vocab)
vocab["<unk>"] = len(vocab)

idx_to_char = {i: c for c, i in vocab.items()}
blank_id = vocab["<blank>"]

# =========================
# LOAD MODEL
# =========================
model = CTCTransformer(INPUT_DIM, len(vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =========================
# DECODE
# =========================
def decode(logits):
    pred = logits.argmax(-1)[0].cpu().numpy()

    result = []
    prev = None

    for p in pred:
        if p != prev and p != blank_id:
            result.append(p)
        prev = p

    chars = [idx_to_char[i] for i in result if i in idx_to_char]
    return "".join(chars)


# =========================
# TEST
# =========================
X = np.load(BASE_PATH / "X_test_face_4w_balanced_norm.npy")

with torch.no_grad():
    for i in [0, 1, 2, 10, 20]:
        x = torch.tensor(X[i]).unsqueeze(0).to(DEVICE)
        logits = model(x)

        print("\nSample", i)
        print("Prediction:", decode(logits))