import numpy as np
from pathlib import Path

# Root der student_grasps_v1 Struktur
ROOT = Path("/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1")

# Output-Ordner eine Ebene höher in "studentGrasping"
OUT = ROOT.parent / "processed_scores"
OUT.mkdir(exist_ok=True)
print("Output Ordner:", OUT)

def build_output_name(npz_path: Path):
    """
    Erzeugt neuen Dateinamen:

    <category>_<model>_<index>.npz
    Beispiel:
    .../02747177/1c3cf618.../0/recording.npz
    → 02747177_1c3cf618..._0.npz
    """

    category = npz_path.parts[-4]  # Ordner z.B. 02747177
    model = npz_path.parts[-3]     # Ordner z.B. 1c3cf618...
    idx = npz_path.parts[-2]       # Ordner z.B. 0

    return f"{category}_{model}_{idx}.npz"

def process_npz(npz_path: Path):
    print(f"\nProcessing: {npz_path}")

    data = np.load(npz_path)

    # sicherstellen dass nur diese keys benutzt werden
    if not {"scores", "grasps"}.issubset(data.files):
        print("  -> ERROR: Datei hat nicht die Keys ['scores', 'grasps']. Übersprungen.")
        return

    grasps = data["grasps"]
    scores = data["scores"]

    print("shapes:", grasps.shape, scores.shape)

    # Filter: Scores > 0
    mask = scores > 0
    grasps_new = grasps[mask]
    scores_new = scores[mask]

    if grasps_new.shape[0] == 0:
        print("  -> Keine Griffe mit Score > 0. Übersprungen.")
        return

    # Neuen Dateinamen bauen
    out_name = build_output_name(npz_path)
    out_path = OUT / out_name

    # Speichern
    np.savez(out_path, grasps=grasps_new, scores=scores_new)
    print(f"  -> gespeichert als {out_path}")

def collect_npz_files(root: Path):
    return list(root.rglob("recording.npz"))

def main():
    npz_files = collect_npz_files(ROOT)
    print(f"Gefundene recording.npz Dateien: {len(npz_files)}")

    for f in npz_files:
        process_npz(f)

if __name__ == "__main__":
    main()