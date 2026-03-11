import pandas as pd
from pathlib import Path

labels_path = r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset\labels\test.csv"
json_folder = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset\test\json")

df = pd.read_csv(labels_path, sep="\t")

# clean text columns just in case
df["SENTENCE_NAME"] = df["SENTENCE_NAME"].astype(str).str.strip()
df["SENTENCE"] = df["SENTENCE"].astype(str).str.strip()

print("Columns:", df.columns.tolist())
print("Total labels:", len(df))

clip_folder = next(json_folder.iterdir())
clip_name = clip_folder.name.strip()

print("Example folder:", repr(clip_name))
print("First 5 SENTENCE_NAME values:")
for x in df["SENTENCE_NAME"].head(5):
    print(repr(x))

# exact match
match = df[df["SENTENCE_NAME"] == clip_name]

print("Matches found:", len(match))

if len(match) > 0:
    print("Sentence:", match.iloc[0]["SENTENCE"])
else:
    print("No exact match found.")

    # optional debug: show rows containing the video id part
    video_id_part = clip_name.split("-")[0] if False else clip_name.split("_")[0]
    print("Video ID part guess:", repr(video_id_part))

    partial = df[df["SENTENCE_NAME"].str.contains(clip_name.split("_")[0], regex=False, na=False)]
    print("Partial matches found:", len(partial))

    if len(partial) > 0:
        print(partial[["SENTENCE_NAME", "SENTENCE"]].head(10).to_string(index=False))