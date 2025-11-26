import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
from functools import partial
import mesh2sdf
import json
from pathlib import Path

# ---- CONFIGURATION ----
CONFIG = {
    # Input: Your raw data folder
    "input_dir": "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1",
    
    # Output: Where the processed files will go
    "output_dir": "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_data_final",
    
    # Resolution: 64 is safer for global scaling (prevents small objects from disappearing)
    "resolution": 32,           
    "padding": 0.1,             # 10% buffer space in the voxel grid
    "num_workers": 8,          # Parallel workers
    "overwrite": True,          # Re-process even if file exists
    
    # Optional: Set this if you want to force a specific scale (e.g. 1/0.5 = 2.0)
    # Otherwise, it is calculated automatically.
    "manual_scale": None 
}

def find_global_scale(root_dir, padding):
    """
    Scans the entire dataset to find the largest object.
    This ensures consistent scaling across all objects.
    """
    print("Step 1: Scanning dataset dimensions to find Global Scale...")
    max_extent = 0.0
    
    # Use a generator to find files to save memory
    files = list(Path(root_dir).rglob("mesh.obj"))
    
    # Sample 1000 random files if dataset is huge to save time, or scan all for accuracy
    # Here we scan all because precision matters for the "biggest" object.
    for f in tqdm(files, desc="Scanning sizes"):
        try:
            # We use trimesh to get bounds
            mesh = trimesh.load(str(f), force='mesh')
            # Extent = Diagonal length of bounding box or Max Dimension length
            # We use Max Dimension Length (e.g. height of a tall lamp)
            extent = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
            # Ignore objects larger than 2 meters (likely noise/artifacts)
            if extent > 2.0: 
                continue
            if extent > max_extent:
                max_extent = extent
        except:
            pass
            
    print(f"\nLargest object size found: {max_extent:.4f} meters")
    
    if max_extent == 0:
        print("Error: Max extent is 0. Defaulting to 1.0")
        max_extent = 1.0

    # Calculate Scale Factor
    # Target: Fit largest object into (1.0 - padding)
    global_scale = (1.0 - padding) / max_extent
    print(f"Global Scale Factor calculated: {global_scale:.4f}")
    print("This scale will be applied to ALL objects.\n")
    return global_scale

def process_trial(config, global_scale, trial_path):
    """
    Worker function to process one trial folder.
    """
    try:
        trial_path = Path(trial_path)
        mesh_path = trial_path / "mesh.obj"
        npz_path = trial_path / "recording.npz"
        
        # Create unique ID from folder structure
        parts = trial_path.parts
        unique_id = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        output_path = Path(config["output_dir"]) / f"{unique_id}.npz"

        if not config["overwrite"] and output_path.exists():
            return True

        if not mesh_path.exists(): return False

        # 1. Load Mesh
        mesh = trimesh.load(str(mesh_path), force='mesh')
        if mesh.vertices.shape[0] == 0: return False
        
        # --- MATH STEP A: Calculate Local Center ---
        # We center every object so it sits at (0,0,0) in the voxel grid.
        bbmin = mesh.vertices.min(0)
        bbmax = mesh.vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        
        # --- MATH STEP B: Transform Mesh ---
        # (Vertices - Center) * Global_Scale
        # This moves the "balloon" down to 0 and scales it consistently.
        norm_vertices = (mesh.vertices - center) * global_scale

        # 2. Compute Voxel Grid (SDF)
        # Level factor determines the "thickness" of the surface in SDF
        level = 2.0 / config["resolution"]
        sdf = mesh2sdf.compute(
            norm_vertices, 
            mesh.faces, 
            config["resolution"], 
            fix=True, 
            level=level, 
            return_mesh=False
        )
        
        # 3. Load & Transform Grasps
        if not npz_path.exists(): return False
        data = np.load(npz_path)
        grasps = data["grasps"]
        scores = data["scores"] if "scores" in data else None
        
        # Filter for Success
        if scores is not None:
            mask = scores > 4
            grasps = grasps[mask]
            scores = scores[mask]
            
        if len(grasps) == 0: return False

        # --- MATH STEP C: Transform Grasps ---
        # Apply the EXACT SAME transformation to the grasp positions.
        # Indices 0,1,2 are X,Y,Z position.
        grasps_fixed = grasps.copy()
        grasps_fixed[:, 0:3] = (grasps[:, 0:3] - center) * global_scale

        # 4. Save Everything
        # We save 'center' and 'scale' so we can reverse this later!
        np.savez_compressed(
            output_path,
            voxel_sdf=sdf.astype(np.float16),       # The Input
            grasps=grasps_fixed.astype(np.float32), # The Target
            scores=scores,
            center=center,                          # Metadata for Inference
            scale=global_scale                      # Metadata for Inference
        )
        return True

    except Exception:
        # Return False if anything crashes (corrupt file, etc.)
        return False

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    root = Path(CONFIG["input_dir"])
    
    # 1. Find Global Scale
    if CONFIG["manual_scale"]:
        global_scale = CONFIG["manual_scale"]
    else:
        global_scale = find_global_scale(root, CONFIG["padding"])
        
    # 2. Find all trials
    print("Scanning for trials...")
    all_npz = list(root.rglob("recording.npz"))
    trial_folders = [p.parent for p in all_npz]
    
    print(f"Processing {len(trial_folders)} trials with Scale {global_scale:.2f}...")
    
    # 3. Run Parallel Processing
    # We use partial to pass the config and scale to every worker
    process_func = partial(process_trial, CONFIG, global_scale)
    
    with multiprocessing.Pool(CONFIG["num_workers"]) as pool:
        # list() forces the iterator to run immediately so we see the progress bar
        list(tqdm(pool.imap_unordered(process_func, trial_folders), total=len(trial_folders)))

    print("\nProcessing Complete.")
    print(f"Data saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()