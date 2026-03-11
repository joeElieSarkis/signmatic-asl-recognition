from pathlib import Path
import numpy as np

base_path = Path(r"C:\Users\Joe\OneDrive\Desktop\signmatic_thesis\dataset\master_thesis_dataset")
json_root = base_path / "train" / "json"

lengths = []

clip_folders = sorted([p for p in json_root.iterdir() if p.is_dir()])

for clip_folder in clip_folders[:1000]:  # sample first 1000
    frame_count = len(list(clip_folder.glob("*.json")))
    lengths.append(frame_count)

lengths = np.array(lengths)

print("Min frames:", lengths.min())
print("Max frames:", lengths.max())
print("Mean frames:", lengths.mean())
print("Median frames:", np.median(lengths))