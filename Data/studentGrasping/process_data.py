import trimesh
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import mesh2sdf
from pathlib import Path

# Global scale of our dataset is 2.5906344993529746

# ---- CONFIGURATION ----
CONFIG = {
    "input_dir": "student_grasps_v1",
    "output_dir": "processed_data_new",
    "resolution": 32,           # 32x32x32 grid
    "padding": 0.1,             # 10% buffer
    "num_workers": 8,           
    "overwrite": True,
    "grasp_threshold": 3
}
# Hand joint range
finger_min = [-0.523598, -0.349065, -0.174532]
finger_max = [ 0.523598,  1.500983,  1.832595]
joints_min = np.array(finger_min * 4, dtype=np.float32)
joints_max = np.array(finger_max * 4, dtype=np.float32)

def find_global_scale(root_dir, padding):
    max_extent = 0.0
    
    files = list(Path(root_dir).rglob("mesh.obj"))
    print(f"Found {len(files)} files")
    
    for f in tqdm(files, desc="Scanning mesh sizes"):
        try:
            mesh = trimesh.load(str(f), force='mesh')
            extent = (mesh.vertices.max(0)-mesh.vertices.min(0)).max()
            
            # Safety if object > 2m
            if extent > 2.0: continue 
            if extent > max_extent: max_extent = extent
            pass
        except:
            pass

    print(f"Largest object: {max_extent:.4f}m")

    # We want (max_extent * scale) = (1.0 - padding)
    global_scale = (1.0 - padding)/max_extent
    print(f"Global Scale Factor calculated: {global_scale:.4f}")
    print("This scale will be applied to ALL objects.\n")
    
    return global_scale

def process_trial(config, global_scale, trial_path):
    """
    GOAL: Load raw mesh/grasps -> Center & Scale -> Voxelize -> Save.
    """
    try:
        trial_path = Path(trial_path)
        mesh_path = trial_path / "mesh.obj"
        npz_path = trial_path / "recording.npz"
        
        # Unique ID generation
        parts = trial_path.parts
        unique_id = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        output_path = Path(config["output_dir"]) / f"{unique_id}.npz"
        
        if not config["overwrite"] and output_path.exists():
            print(f"Output Path{output_path}exists")
            return True
        if not mesh_path.exists(): return False

        mesh = trimesh.load(str(mesh_path), force='mesh')
        bbmin = mesh.vertices.min(axis=0)
        bbmax = mesh.vertices.max(axis=0)
        center = (bbmax + bbmin) * 0.5
        
        # Formula: new_pos = (old_pos - center) * scale
        norm_vertices = (mesh.vertices - center) * global_scale

        # The 'level' controls the surface thickness definition.
        level = 2.0 / config["resolution"]
        
        sdf = mesh2sdf.compute(
            norm_vertices,
            mesh.faces,
            CONFIG["resolution"],
            fix=True,
            level=level,
            return_mesh=False)
        
        # Load Grasps
        if not npz_path.exists(): return False
        data = np.load(npz_path)
        grasps = data["grasps"]
        scores = data["scores"]
        
        # Filter
        mask = scores > CONFIG['grasp_threshold']
        grasps = grasps[mask]
        scores = scores[mask]
        if len(grasps) == 0: return False

        # Apply the EXACT SAME math to the grasp positions.
        grasps_fixed = grasps.copy()
        grasps_fixed[:, 0:3] = (grasps[:, 0:3] - center) * global_scale

        # Normalize joint angles [-1, 1]
        grasps_fixed[:, 7:] = 2 * (grasps_fixed[:, 7:] - joints_min) / (joints_max - joints_min) - 1

        
        np.savez_compressed(
            output_path,
            voxel_sdf=sdf.astype(np.float16),
            grasps=grasps_fixed.astype(np.float32),
            center=center, # Save metadata to reverse later!
            scale=global_scale
        )
        return True

    except Exception as e:
        print(f"FAILED {unique_id}: {e}")
        return False

def main():
    # Setup directories
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    CONFIG["output_dir"] = str(root / CONFIG['output_dir'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    
    scale = find_global_scale(root, CONFIG["padding"])
    print(f"Global Scale: {scale}")
    
    # Get all Files
    all_npz = list(root.rglob("recording.npz"))
    folders = [p.parent for p in all_npz]
    
    # Parallel Execution
    process_func = partial(process_trial, CONFIG, scale)
    with multiprocessing.Pool(CONFIG["num_workers"]) as pool:
        list(tqdm(pool.imap_unordered(process_func, folders), total=len(folders)))
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()