#run with: python Testing/mesh_to_voxel_testset.py
import trimesh
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import mesh2sdf
from pathlib import Path

# Global scale of test dataset is 1.8111312124316046

# ---- CONFIGURATION ----
CONFIG = {
    "input_dir": ".\Data\Testset\MultiGripperGrasp",
    "output_dir": ".\Data\Testset\Processed_Data_MultiGripperGrasp",
    "resolution": 32,           # 32x32x32 grid
    "padding": 0.1,             # 10% buffer
    "num_workers": 8,           
    "overwrite": True,
    "grasp_threshold": 3
}
def center_and_scale_meshes(root_dir, scale_factor=2):
    """
    Centers all meshes in the dataset:
      - X/Y midpoint set to 0
      - Bottom of the object (min Z) set to 0
    Optionally scales the mesh by 'scale_factor'.
    Saves the centered (and scaled) mesh as 'model_centered.obj' in the same folder.
    """
    files = list(Path(root_dir).rglob("model.obj"))
    print(f"Found {len(files)} meshes to center and scale")

    for f in tqdm(files, desc="Centering and scaling meshes"):
        try:
            mesh = trimesh.load(str(f), force='mesh')
            """"
            # Compute bounding box
            bb_min = mesh.vertices.min(axis=0)
            bb_max = mesh.vertices.max(axis=0)
            
            # X/Y center
            center_xy = (bb_max[:2] + bb_min[:2]) * 0.5
            # Z bottom
            z_min = bb_min[2]
            
            # Compute offset
            offset = np.array([center_xy[0], center_xy[1], z_min])
            
            # Shift vertices to center
            mesh.vertices -= offset
            
            # Scale mesh
            mesh.vertices *= scale_factor
            """
            # Save centered and scaled mesh in the same folder
            centered_path = f"{f.parent}/model_centered.obj"
            mesh.export(centered_path)
        
        except Exception as e:
            print(f"Failed to process {f}: {e}")


def find_global_scale(root_dir, padding):
    max_extent = 0.0
    
    files = list(Path(root_dir).rglob("model_centered.obj"))
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
        mesh_path = trial_path / "model_centered.obj"
        
        # Unique ID generation
        parts = trial_path.parts
        unique_id = parts[-2]
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
        
        np.savez_compressed(
            output_path,
            voxel_sdf=sdf.astype(np.float16),
            center=center, # Save metadata to reverse later!
            scale=global_scale
        )
        return True

    except Exception as e:
        print(f"FAILED {unique_id}: {e}")
        return False

def main():
    # Setup directories
    input_root = Path(CONFIG["input_dir"])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    #center meshes and overwrite old meshes 
    center_and_scale_meshes(input_root)

    # Find global scale
    #scale = find_global_scale(input_root, CONFIG["padding"])
    scale = 2.5906344993529746
    print(f"Global Scale: {scale}")

    # Find all folders that contain model_centered.obj
    all_meshes = list(input_root.rglob("model_centered.obj"))
    folders = [p.parent for p in all_meshes]

    print(f"Found {len(folders)} meshes")

    # Parallel Execution
    process_func = partial(process_trial, CONFIG, scale)
    with multiprocessing.Pool(CONFIG["num_workers"]) as pool:
        list(tqdm(pool.imap_unordered(process_func, folders), total=len(folders)))

    print("\nProcessing Complete.")


if __name__ == "__main__":
    main()