from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# =========================
# Paths
# =========================
BASE_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words\data")
OUTPUT_PATH = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\experiments\face_4words_balanced\data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Keep only phrases that appear at least this many times
MIN_FREQ = 2

# Cap how many samples per phrase we keep
MAX_PER_LABEL_TRAIN = 50
MAX_PER_LABEL_VAL = 20
MAX_PER_LABEL_TEST = 20


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def save_labels(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")


def process_split(split, max_per_label):
    X = np.load(BASE_PATH / f"X_{split}_face_4w.npy")
    y = load_labels(BASE_PATH / f"y_{split}_face_4w.txt")
    clip_names = load_labels(BASE_PATH / f"clip_names_{split}_face_4w.txt")

    assert len(X) == len(y) == len(clip_names), f"Mismatch in {split}"

    counts = Counter(y)

    # Keep only labels that appear enough times
    valid_labels = {label for label, cnt in counts.items() if cnt >= MIN_FREQ}

    kept_X = []
    kept_y = []
    kept_clip_names = []

    used_per_label = defaultdict(int)

    for x_item, y_item, clip_name in zip(X, y, clip_names):
        if y_item not in valid_labels:
            continue

        if used_per_label[y_item] >= max_per_label:
            continue

        kept_X.append(x_item)
        kept_y.append(y_item)
        kept_clip_names.append(clip_name)
        used_per_label[y_item] += 1

    kept_X = np.array(kept_X, dtype=np.float32)

    np.save(OUTPUT_PATH / f"X_{split}_face_4w_balanced.npy", kept_X)
    save_labels(OUTPUT_PATH / f"y_{split}_face_4w_balanced.txt", kept_y)
    save_labels(OUTPUT_PATH / f"clip_names_{split}_face_4w_balanced.txt", kept_clip_names)

    print(f"\n{split} balanced subset done.")
    print("X shape:", kept_X.shape)
    print("Labels:", len(kept_y))

    final_counts = Counter(kept_y)
    print("Unique labels:", len(final_counts))
    print("Top 20 labels:")
    for label, cnt in final_counts.most_common(20):
        print(f"{label}: {cnt}")


if __name__ == "__main__":
    process_split("train", MAX_PER_LABEL_TRAIN)
    process_split("val", MAX_PER_LABEL_VAL)
    process_split("test", MAX_PER_LABEL_TEST)