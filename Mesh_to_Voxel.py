import os
import time
import trimesh
import numpy as np
import mesh2sdf

# -------------------------------
# Settings
# -------------------------------
mesh_folder = "Data/studentGrasping/student_grasps_v1"
output_folder = "Data/studentGrasping/student_grasps_v1_processed"
os.makedirs(output_folder, exist_ok=True)

# --- This is a dictionary, so we will access it with config['key'] ---
config = {
    "mesh_scale": 1.0,   # Target range [-1, 1]
    "sdf_size": 128,     # SDF grid resolution
    "max_objects": 3,    # For testing: limit number of instances
    "mesh_folder": mesh_folder,
    "output_folder": output_folder,
}
config["level"] = 2.0 / config["sdf_size"]  # Step size

# -------------------------------
# 2. Collect all meshes recursively in sorted order
# -------------------------------
mesh_paths = []
print("Collecting mesh paths...")
# --- FIX: Use dictionary access config['mesh_folder'] ---
for category in sorted(os.listdir(config['mesh_folder'])):
    category_path = os.path.join(config['mesh_folder'], category)
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

if config['max_objects'] is not None:
    print(f"Limiting to {config['max_objects']} meshes for testing.")
    mesh_paths = mesh_paths[:config['max_objects']]

print(f"Found {len(mesh_paths)} meshes to process.")

# -------------------------------
# 3. Load all meshes, find max bounding box
# -------------------------------
print("Loading all meshes to find max bounding box...")
mesh_data = []
max_size = 0.0
total_load_time = 0.0

for filepath in mesh_paths:
    t_start = time.time()
    try:
        mesh = trimesh.load(filepath, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            print(f"Skipping {filepath}: Not a valid mesh.")
            continue
            
        vertices = mesh.vertices
        bbox_min = vertices.min(0)
        bbox_max = vertices.max(0)
        center = (bbox_max + bbox_min) / 2.0
        
        bbox_size = (bbox_max - bbox_min).max()
        if bbox_size > max_size:
            max_size = bbox_size
        
        mesh_data.append({'mesh': mesh, 'center': center, 'path': filepath})
        total_load_time += (time.time() - t_start)
        
    except Exception as e:
        print(f"Skipping {filepath}: Failed to load. Error: {e}")

print(f"Loaded {len(mesh_data)} valid meshes in {total_load_time:.2f}s.")

# -------------------------------
# 4. Process meshes: center, scale, compute SDF, save
# -------------------------------
if not mesh_data or max_size == 0.0:
    print("No valid meshes were loaded. Exiting.")
else:
    # --- FIX: Use dictionary access ---
    scale_factor = 2.0 * config['mesh_scale'] / max_size
    print(f"\nMaximum bounding box: {max_size:.4f}")
    print(f"Uniform scale factor: {scale_factor:.4f}")
    print(f"\nProcessing {len(mesh_data)} meshes...")

    total_process_time = 0.0
    for data in mesh_data:
        t_start_proc = time.time()
        
        mesh = data['mesh']
        center = data['center']
        filepath = data['path']
        
        fname = os.path.basename(filepath)
        # --- FIX: Use dictionary access ---
        rel_path = os.path.relpath(os.path.dirname(filepath), config['mesh_folder'])
        out_dir = os.path.join(config['output_folder'], rel_path)
        os.makedirs(out_dir, exist_ok=True)

        vertices_scaled = (mesh.vertices - center) * scale_factor

        try:
            # --- FIX: Use dictionary access ---
            sdf, mesh_fixed = mesh2sdf.compute(
                vertices_scaled, mesh.faces, size=config['sdf_size'], fix=True, level=config['level'], return_mesh=True
            )
            
            mesh_fixed.vertices = (mesh_fixed.vertices / scale_factor) + center

            output_name = os.path.splitext(fname)[0]
            np.save(os.path.join(out_dir, f"{output_name}.npy"), sdf)
            mesh_fixed.export(os.path.join(out_dir, f"{output_name}.fixed.obj"))
            
            t_end_proc = time.time()
            process_time = t_end_proc - t_start_proc
            total_process_time += process_time
            
            if (mesh_data.index(data) + 1) % 50 == 0: # Print update every 50
                print(f"Processed {mesh_data.index(data) + 1}/{len(mesh_data)} meshes...")
            
        except Exception as e:
            print(f"Failed to compute SDF for {fname}. Error: {e}")

    print("\nAll selected meshes have been scaled and SDFs computed.")