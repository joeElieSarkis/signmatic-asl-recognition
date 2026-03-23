import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced_normalized\data")
MODEL_OUTPUT = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced_normalized\models")
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
MAX_TOKENS = 6           # <=4 words + <sos> + <eos>
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.3
INPUT_DIM = 411
SEQ_LEN = 60
PATIENCE = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def tokenize_sentence(sentence, vocab):
    """
    Convert sentence to token IDs.
    Example:
    'thank you' -> [<sos>, thank, you, <eos>, <pad>, <pad>]
    """
    tokens = ["<sos>"] + sentence.split() + ["<eos>"]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    if len(ids) < MAX_TOKENS:
        ids += [vocab["<pad>"]] * (MAX_TOKENS - len(ids))
    else:
        ids = ids[:MAX_TOKENS]

    return ids


class SignDataset(Dataset):
    def __init__(self, x_path, y_path, vocab):
        self.X = np.load(x_path)
        self.y_text = load_labels(y_path)
        self.y_ids = [tokenize_sentence(sent, vocab) for sent in self.y_text]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y_ids[idx], dtype=torch.long)
        return x, y


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer inputs.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerSignModel(nn.Module):
    """
    Transformer encoder model:
    (batch, 60, 411) -> fixed token sequence
    """
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_dim, dropout, vocab_size, max_tokens):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, vocab_size * max_tokens)

        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        x = self.fc(x)
        x = x.view(-1, self.max_tokens, self.vocab_size)
        return x


# Load vocab
with open(BASE_PATH / "vocab_face_4w_balanced_norm.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

# Datasets
train_dataset = SignDataset(
    BASE_PATH / "X_train_face_4w_balanced_norm.npy",
    BASE_PATH / "y_train_face_4w_balanced_norm.txt",
    vocab
)

val_dataset = SignDataset(
    BASE_PATH / "X_val_face_4w_balanced_norm.npy",
    BASE_PATH / "y_val_face_4w_balanced_norm.txt",
    vocab
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = TransformerSignModel(
    input_dim=INPUT_DIM,
    d_model=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    vocab_size=vocab_size,
    max_tokens=MAX_TOKENS
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
epochs_without_improvement = 0

print(f"Using device: {DEVICE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Vocab size: {vocab_size}")

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                y_batch.reshape(-1)
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_OUTPUT / "best_face_4w_balanced_norm_transformer.pt")
        print("Saved new best model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training finished.")
print("Best validation loss:", best_val_loss)