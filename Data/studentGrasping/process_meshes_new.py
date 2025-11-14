import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm
import traceback
import multiprocessing
from functools import partial
import time
import mesh2sdf

# ---
# This script is now based on the official mesh2sdf pipeline logic
# provided by the user. It replaces all previous attempts.
# ---


def process_file(config, mesh_path_tuple):
    """
    Runs the full processing pipeline for a single file,
    using the user-provided mesh2sdf logic.
    """
    
    # Unpack tuple from multiprocessing
    input_file, output_base = mesh_path_tuple
    
    OUTPUT_NPY_FILE = output_base + ".npy"
    OUTPUT_FIXED_MESH = output_base + ".fixed.obj"
    
    try:
        # --- Config from user's script ---
        mesh_scale = config["mesh_scale"]
        size = config["resolution"]
        level = config["level_factor"] / size # e.g., 2 / 128
        
        # --- 1. Load mesh ---
        mesh = trimesh.load(input_file, force='mesh')

        # --- 2. Normalize mesh (using user's exact logic) ---
        vertices = mesh.vertices
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale_factor = 2.0 * mesh_scale / (bbmax - bbmin).max()
        norm_vertices = (vertices - center) * scale_factor

        # --- 3. Compute SDF ---
        # This is the core call.
        # fix=True: Heals the mesh (like we tried to do manually)
        # return_mesh=True: Returns the fixed mesh
        sdf, fixed_mesh_norm = mesh2sdf.compute(
            norm_vertices, 
            mesh.faces, 
            size, 
            fix=True, 
            level=level, 
            return_mesh=True
        )

        # --- 4. Save .npy file ---
        # We save the raw SDF, as per the user's script
        np.save(OUTPUT_NPY_FILE, sdf.astype(np.float32))

        # --- 5. Save fixed mesh (optional) ---
        if config["save_fixed_mesh"]:
            # Un-normalize the vertices of the *fixed* mesh to
            # save it in its original position and scale.
            fixed_mesh_norm.vertices = fixed_mesh_norm.vertices / scale_factor + center
            fixed_mesh_norm.export(OUTPUT_FIXED_MESH)

        return True, input_file, None

    except Exception:
        # Catch all errors to prevent crashing the pool
        error_msg = traceback.format_exc()
        return False, input_file, error_msg


# ---
# NEW Main Execution Block
# (Re-enabled multiprocessing)
# ---

if __name__ == "__main__":
    
    # ---
    # MAIN DATASET CONFIGURATION
    # ---
    DATASET_CONFIG = {
        "input_dir": "student_grasps_v1",
        "output_folder": "processed_meshes",
        "limit": 2,               # Set to a number (e.g., 10) to test
        "overwrite": True,          # Set True to re-process existing files
        "save_fixed_mesh": False,    # Set True to save the .fixed.obj files
        
        # Parameters from the mesh2sdf script
        "resolution": 32,           # Voxel grid resolution (was 'size')
        "mesh_scale": 0.8,           # Scale for normalization
        "level_factor": 2,           # Numerator for 'level' (level = 2 / 128)

        # Multiprocessing
        "num_workers": 5            # Number of parallel processes
    }
    
    # --- Step 1: Create Output Directory ---
    output_dir = DATASET_CONFIG["output_folder"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 2: Find all 'mesh.obj' files ---
    print("Finding mesh files...")
    search_path = os.path.join(DATASET_CONFIG["input_dir"], "*", "*", "*", "mesh.obj")
    all_mesh_files = glob.glob(search_path)

    valid_mesh_files = []
    for f in all_mesh_files:
        parts = f.split(os.sep)
        if len(parts) >= 2 and parts[-2].isdigit():
            valid_mesh_files.append(f)
    
    print(f"Found {len(valid_mesh_files)} valid mesh files.")

    # --- Step 3: Create Task List ---
    tasks = []
    for mesh_path in valid_mesh_files:
        try:
            parts = mesh_path.split(os.sep)
            category_id = parts[-4]
            instance_id = parts[-3]
            integer_id = parts[-2]
        except IndexError:
            print(f"Warning: Could not parse path {mesh_path}. Skipping.")
            continue
            
        base_name = f"{category_id}_{instance_id}_{integer_id}"
        output_base = os.path.join(output_dir, base_name)
        
        # Check for overwrite
        output_npy = output_base + ".npy"
        if not DATASET_CONFIG["overwrite"] and os.path.exists(output_npy):
            continue
            
        # Add a tuple of (input_file, output_base) to the task list
        tasks.append((mesh_path, output_base))
            
    # --- Step 4: Apply Limit ---
    if DATASET_CONFIG["limit"] is not None and DATASET_CONFIG["limit"] > 0:
        print(f"Limiting processing to {DATASET_CONFIG['limit']} files.")
        tasks = tasks[:DATASET_CONFIG["limit"]]

    print(f"Found {len(tasks)} new or non-processed files.")
    
    if len(tasks) == 0:
        print("No files to process. Exiting.")
        exit()

    # --- Step 5: Process Files (in Parallel) ---
    num_workers = DATASET_CONFIG["num_workers"]
    print(f"Starting processing pool with {num_workers} workers...")
    
    # Create a partial function to pass the fixed DATASET_CONFIG
    processor_func = partial(process_file, DATASET_CONFIG)
    
    success_count = 0
    failed_files = []
    
    with multiprocessing.Pool(num_workers) as pool:
        # Wrap the imap results with tqdm for a progress bar
        # pool.imap_unordered is fast as it doesn't wait for order
        results_iter = pool.imap_unordered(processor_func, tasks)
        
        for result in tqdm(results_iter, total=len(tasks), desc="Processing Meshes"):
            success, path, error = result
            if success:
                success_count += 1
            else:
                failed_files.append((path, error))

    # --- Step 6: Print Summary ---
    print("\n---")
    print("Dataset Processing Complete!")
    print(f"  {success_count} files processed successfully.")
    print(f"  {len(failed_files)} files failed.")
    
    if failed_files:
        print("\nFailed files:")
        for path, error in failed_files:
            error_line = str(error).split('\n')[-2] if error else "Unknown"
            print(f"Failed: {path} | Error: {error_line}")