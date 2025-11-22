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


def process_file(config, mesh_path_tuple):
    """
    Runs the full processing pipeline for a single file,
    using the "Global Scale, Local Center" method.
    
    This keeps the relative sizing (from global_scale_factor)
    but centers each object in the grid (from local_center)
    for easier learning.
    """
    
    # Unpack tuple from multiprocessing
    input_file, output_base = mesh_path_tuple
    
    OUTPUT_NPY_FILE = output_base + ".npy"
    OUTPUT_FIXED_MESH = output_base + ".fixed.obj"
    
    try:
        # We get the GLOBAL scale factor from the main config
        global_scale_factor = config["global_scale_factor"]
        
        size = config["resolution"]
        level = config["level_factor"] / size # e.g., 2 / 128
        
        mesh = trimesh.load(input_file, force='mesh')

        vertices = mesh.vertices
        
        # Calculate the local center for *this* mesh
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        local_center = (bbmin + bbmax) * 0.5
        
        # Normalize using the local center but the global scale
        norm_vertices = (vertices - local_center) * global_scale_factor
        # ---

        # Compute SDF ---
        sdf, fixed_mesh_norm = mesh2sdf.compute(
            norm_vertices, 
            mesh.faces, 
            size, 
            fix=True, 
            level=level, 
            return_mesh=True
        )

        # We save the raw SDF
        np.save(OUTPUT_NPY_FILE, sdf.astype(np.float32))

        # Save fixed mesh (optional) ---
        if config["save_fixed_mesh"]:
            fixed_mesh_norm.vertices = fixed_mesh_norm.vertices / global_scale_factor + local_center
            fixed_mesh_norm.export(OUTPUT_FIXED_MESH)

        return True, input_file, None

    except Exception:
        # Catch all errors to prevent crashing the pool
        error_msg = traceback.format_exc()
        return False, input_file, error_msg


if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    DATASET_CONFIG = {
        "input_dir": os.path.join(script_dir, "student_grasps_v1"),
        "output_folder": os.path.join(script_dir, "processed_meshes"),
        "limit": None,            # Set to a number (e.g., 10) to test
        "overwrite": True,        # Set True to re-process existing files
        "save_fixed_mesh": False,   # Set True to save the .fixed.obj files
        
        # Parameters from the mesh2sdf script
        "resolution": 64,         # Voxel grid resolution (was 'size')
        "mesh_scale": 0.8,        # Scale for normalization
        "level_factor": 5,        # Numerator for 'level' (level = 2 / 128)

        # Multiprocessing
        "num_workers": 10           # Number of parallel processes
    }
    
    # Create Output Directory ---
    output_dir = DATASET_CONFIG["output_folder"]
    os.makedirs(output_dir, exist_ok=True)

    # Find all 'mesh.obj' files ---
    print("Finding mesh files...")
    search_path = os.path.join(DATASET_CONFIG["input_dir"], "*", "*", "*", "mesh.obj")
    all_mesh_files = glob.glob(search_path)

    valid_mesh_files = []
    for f in all_mesh_files:
        parts = f.split(os.sep)
        if len(parts) >= 2 and parts[-2].isdigit():
            valid_mesh_files.append(f)
    
    print(f"Found {len(valid_mesh_files)} valid mesh files.")
    
    if not valid_mesh_files:
        print("No valid files found. Exiting.")
        exit()

    # Pre-calculation of Global Bounding Box
    print("Calculating global bounding box for all meshes...")
    global_bbmin = np.array([np.inf, np.inf, np.inf])
    global_bbmax = np.array([-np.inf, -np.inf, -np.inf])

    for mesh_path in tqdm(valid_mesh_files, desc="Finding Global Bounds"):
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if mesh.vertices.shape[0] == 0:
                print(f"\nWarning: {mesh_path} has 0 vertices. Skipping.")
                continue
            global_bbmin = np.minimum(global_bbmin, mesh.vertices.min(0))
            global_bbmax = np.maximum(global_bbmax, mesh.vertices.max(0))
        except Exception as e:
            print(f"\nWarning: Could not load {mesh_path} for bounds calculation. Skipping. Error: {e}")
    
    # Calculate global center and scale factor
    global_center = (global_bbmin + global_bbmax) * 0.5
    global_max_dim = (global_bbmax - global_bbmin).max()
    
    if global_max_dim == 0:
        print("Warning: Global max dimension is 0. All meshes are points? Setting to 1.0 to avoid errors.")
        global_max_dim = 1.0 
        
    global_scale_factor = 2.0 * DATASET_CONFIG["mesh_scale"] / global_max_dim

    # PRINT FOR OUTLIER CHECK
    print("\n" + "="*30)
    print("GLOBAL BOUNDS CHECK (FOR OUTLIERS)")
    print(f"  Global Bounding Box Min: {global_bbmin}")
    print(f"  Global Bounding Box Max: {global_bbmax}")
    print(f"  Global Max Dimension:    {global_max_dim}")
    print(f"  Global Scale Factor:     {global_scale_factor}")
    print("="*30 + "\n")
    # ---

    # Add global scale factor to the config to be passed to workers ---
    DATASET_CONFIG["global_scale_factor"] = global_scale_factor


    # Create Task List ---
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
            
        tasks.append((mesh_path, output_base))
            
    # Apply Limit
    if DATASET_CONFIG["limit"] is not None and DATASET_CONFIG["limit"] > 0:
        print(f"Limiting processing to {DATASET_CONFIG['limit']} files.")
        tasks = tasks[:DATASET_CONFIG["limit"]]

    print(f"Found {len(tasks)} new or non-processed files.")
    
    if len(tasks) == 0:
        print("No files to process. Exiting.")
        exit()

    # Process Files (in Parallel) ---
    num_workers = DATASET_CONFIG["num_workers"]
    print(f"Starting processing pool with {num_workers} workers...")
    
    # Create a partial function to pass the fixed DATASET_CONFIG
    processor_func = partial(process_file, DATASET_CONFIG)
    
    success_count = 0
    failed_files = []
    
    with multiprocessing.Pool(num_workers) as pool:
        results_iter = pool.imap_unordered(processor_func, tasks)
        
        for result in tqdm(results_iter, total=len(tasks), desc="Processing Meshes"):
            success, path, error = result
            if success:
                success_count += 1
            else:
                failed_files.append((path, error))

    # Print Summary
    print("\n---")
    print("Dataset Processing Complete!")
    print(f"  {success_count} files processed successfully.")
    print(f"  {len(failed_files)} files failed.")
    
    if failed_files:
        print("\nFailed files:")
        for path, error in failed_files:
            error_line = str(error).split('\n')[-2] if error else "Unknown"
            print(f"Failed: {path} | Error: {error_line}")