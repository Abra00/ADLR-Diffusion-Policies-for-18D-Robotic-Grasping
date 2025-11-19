import numpy as np
import trimesh
from skimage.measure import marching_cubes

# -------------------------------
# Settings
# -------------------------------
file_path = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1_processed/02747177/1c3cf618a6790f1021c6005997c63924/0/mesh.npy"

# -------------------------------
# Load SDF
# -------------------------------
sdf = np.load(file_path)
size = sdf.shape[0]  # z.B. 32
level=0


print(f"SDF shape: {sdf.shape}, min={sdf.min():.6f}, max={sdf.max():.6f}")


# -------------------------------
# Marching Cubes
# -------------------------------
vertices, faces, _, _ = marching_cubes(sdf, level=level)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# -------------------------------
# Größte Komponente extrahieren
# -------------------------------
components = mesh.split(only_watertight=False)
bbox_sizes = [(c.vertices.max(0) - c.vertices.min(0)).max() for c in components]
max_component = np.argmax(bbox_sizes)
mesh = components[max_component]

# -------------------------------
# Normalisierung auf [-1, 1] (optional)
# -------------------------------
mesh.vertices = mesh.vertices * (2.0 / size) - 1.0

# -------------------------------
# Anzeigen
# -------------------------------
mesh.show()
