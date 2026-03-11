import pandas as pd
from pathlib import Path

base_path = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")

for split in ["train", "val", "test"]:
    labels_path = base_path / "labels" / f"{split}.csv"
    df = pd.read_csv(labels_path, sep="\t")
    df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()

    print(f"\n--- {split.upper()} ---")
    print("Total samples:", len(df))
    print("Unique sentences:", df["SENTENCE"].nunique())

    sentence_counts = df["SENTENCE"].value_counts()

    print("\nTop 20 most frequent sentences:")
    print(sentence_counts.head(20).to_string())

    short_df = df[df["SENTENCE"].str.split().str.len() <= 3]
    print("\nSamples with sentence length <= 3 words:", len(short_df))
    print("Unique short sentences:", short_df["SENTENCE"].nunique())

    print("\nExample short sentences:")
    print(short_df["SENTENCE"].drop_duplicates().head(20).to_string(index=False))