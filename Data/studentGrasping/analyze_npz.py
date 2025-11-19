import numpy as np
from pathlib import Path

# Pfad zur Datei
data_path = Path("/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_scores/04468005_25c3710ca5f245b8c62b7ed19a553484_0.npz")

# Datei laden
data = np.load(data_path)

for key in data.files:
    arr = data[key]
    print(f"--- {key} ---")
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("Sample:", arr[0] if arr.ndim > 0 else arr)
    print()