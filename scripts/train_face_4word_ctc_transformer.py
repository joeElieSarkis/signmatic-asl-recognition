import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# PATHS 
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced_normalized\data")

MODEL_PATH = BASE_PATH.parent / "models" / "best_face_4w_ctc.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# SETTINGS
# =========================
INPUT_DIM = 411
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 40
LR = 5e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATA
# =========================
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().split() for line in f]


class CTCDataset(Dataset):
    def __init__(self, x_path, y_path, vocab):
        self.X = np.load(x_path)
        self.y = load_labels(y_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        y_ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in self.y[idx]]
        y_ids = torch.tensor(y_ids, dtype=torch.long)

        return x, y_ids, 30, len(y_ids)


def collate_fn(batch):
    xs, ys, input_lens, target_lens = zip(*batch)

    xs = torch.stack(xs)
    ys = torch.cat(ys)

    input_lens = torch.tensor(input_lens, dtype=torch.long)
    target_lens = torch.tensor(target_lens, dtype=torch.long)

    return xs, ys, input_lens, target_lens


# =========================
# MODEL
# =========================
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
            in_channels=D_MODEL,
            out_channels=D_MODEL,
            kernel_size=3,
            stride=2,
            padding=1
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
        x = self.input_proj(x)          # (B, 60, D)

        x = x.transpose(1, 2)           # (B, D, 60)
        x = self.temporal_conv(x)       # (B, D, 30)
        x = x.transpose(1, 2)           # (B, 30, D)

        x = self.pos(x)
        x = self.encoder(x)
        return self.fc(x)


# =========================
# LOAD VOCAB
# =========================
with open(BASE_PATH / "vocab_face_4w_balanced_norm.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# ADD BLANK TOKEN (CTC REQUIRED)
if "<blank>" not in vocab:
    vocab["<blank>"] = len(vocab)

vocab_size = len(vocab)
blank_id = vocab["<blank>"]


# =========================
# DATA LOADERS
# =========================
train_ds = CTCDataset(
    BASE_PATH / "X_train_face_4w_balanced_norm.npy",
    BASE_PATH / "y_train_face_4w_balanced_norm.txt",
    vocab
)

val_ds = CTCDataset(
    BASE_PATH / "X_val_face_4w_balanced_norm.npy",
    BASE_PATH / "y_val_face_4w_balanced_norm.txt",
    vocab
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# =========================
# TRAIN
# =========================
model = CTCTransformer(INPUT_DIM, vocab_size).to(DEVICE)

criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y, input_lens, target_lens in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)
        log_probs = logits.log_softmax(-1).transpose(0, 1)

        loss = criterion(log_probs, y, input_lens, target_lens)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y, input_lens, target_lens in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            log_probs = logits.log_softmax(-1).transpose(0, 1)

            loss = criterion(log_probs, y, input_lens, target_lens)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | train={train_loss:.4f} val={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved best model")