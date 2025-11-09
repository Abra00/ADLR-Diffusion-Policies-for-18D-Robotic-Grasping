import os
import time
import trimesh
import numpy as np
import mesh2sdf

# -------------------------------
# Settings
# -------------------------------
mesh_folder = "student_grasps_v1"   # Root folder containing the dataset
output_folder = "student_grasps_v1_processed"         # Folder to save SDFs and fixed meshes
os.makedirs(output_folder, exist_ok=True)

mesh_scale = 1       # Target range [-1, 1]
size = 128           # SDF grid resolution
level = 2 / size     # Step size for mesh2sdf
max_objects = 2      # For testing: limit number of instances to process (None for all)

# -------------------------------
# 1. Collect all meshes recursively in sorted order
# -------------------------------
mesh_paths = []

for category in sorted(os.listdir(mesh_folder)):
    category_path = os.path.join(mesh_folder, category)
    if not os.path.isdir(category_path):
        continue
    for instance in sorted(os.listdir(category_path)):
        instance_path = os.path.join(category_path, instance)
        if not os.path.isdir(instance_path):
            continue

        trials = sorted([d for d in os.listdir(instance_path) if os.path.isdir(os.path.join(instance_path, d))])
        for trial in trials:
            trial_path = os.path.join(instance_path, trial)
            for fname in os.listdir(trial_path):
                if fname.endswith(".obj") or fname.endswith(".off"):
                    mesh_paths.append(os.path.join(trial_path, fname))

# Limit number of instances for testing
if max_objects is not None:
    mesh_paths = mesh_paths[:max_objects]

# -------------------------------
# 2. Determine largest bounding box to compute uniform scaling
# -------------------------------
max_size = 0.0
for filepath in mesh_paths:
    mesh = trimesh.load(filepath)
    vertices = mesh.vertices
    bbox_size = (vertices.max(0) - vertices.min(0)).max()
    if bbox_size > max_size:
        max_size = bbox_size

scale_factor = 2.0 * mesh_scale / max_size
print(f"Maximum bounding box: {max_size:.4f}")
print(f"Uniform scale factor: {scale_factor:.4f}")

# -------------------------------
# 3. Process meshes: center, scale, compute SDF, save
# -------------------------------
for filepath in mesh_paths:
    fname = os.path.basename(filepath)
    rel_path = os.path.relpath(os.path.dirname(filepath), mesh_folder)
    out_dir = os.path.join(output_folder, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(filepath)
    vertices = mesh.vertices

    # Center and scale mesh
    center = vertices.mean(0)
    vertices = (vertices - center) * scale_factor

    t0 = time.time()
    sdf, mesh_fixed = mesh2sdf.compute(
        vertices, mesh.faces, size=size, fix=True, level=level, return_mesh=True
    )
    t1 = time.time()

    # Optional: transform mesh back to original position
    mesh_fixed.vertices = mesh_fixed.vertices / scale_factor + center

    # Save SDF and fixed mesh
    np.save(os.path.join(out_dir, fname.replace(".obj",".npy").replace(".off",".npy")), sdf)
    mesh_fixed.export(os.path.join(out_dir, fname.replace(".obj",".fixed.obj").replace(".off",".fixed.obj")))

    print(f"Processed {fname} in {t1-t0:.2f}s")

print("All selected meshes have been scaled and SDFs computed.")
