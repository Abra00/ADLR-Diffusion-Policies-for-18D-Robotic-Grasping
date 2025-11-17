import os
import sys
import numpy as np
import skimage.measure
import trimesh

# ---
# CONFIGURATION
# ---
# This parameter MUST match the "mesh_scale" you used
# in 'process_meshes_new.py'
MESH_SCALE = 0.8
# ---

def visualize_interactive(input_npy_file):
    """
    Processes a .npy file and shows an INTERACTIVE 3D view
    of the main surface (level 0.0).
    
    It also saves the other level sets for inspection.
    """
    
    # Define the SDF levels to extract
    # 0.0 is the main surface
    levels = [-0.02, 0.0, 0.02]

    # Create an output folder named after the .npy file
    folder = input_npy_file[:-4] # Remove .npy
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f"Loading {input_npy_file}...")
    print(f"Saving 3D meshes to folder: {folder}/")

    sdf = np.load(input_npy_file)
    size = sdf.shape[0] # Should be 32
    
    # Print the SDF range
    print(f"SDF grid loaded. Shape: {sdf.shape}, Max: {sdf.max():.4f}, Min: {sdf.min():.4f}")

    # Extract 3D Level Sets (Marching Cubes)
    print(f"Extracting 3D surfaces at levels: {levels}...")
    
    main_surface_mesh = None

    for i, level in enumerate(levels):
        # Run marching cubes to find the surface
        vtx, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

        # De-normalize the vertex coordinates
        # We map from grid coords [0, size] to world coords [-MESH_SCALE, +MESH_SCALE]
        vtx_scaled = vtx * (MESH_SCALE * 2.0 / size) - MESH_SCALE
        
        # Create and save the mesh
        mesh = trimesh.Trimesh(vtx_scaled, faces)
        output_obj = os.path.join(folder, 'level_%.2f.obj' % level)
        mesh.export(output_obj)
        print(f"  Saved {output_obj}")

        if level == 0.0:
            # This is the main surface, save it for showing
            main_surface_mesh = mesh

    # Show Interactive 3D View

    if main_surface_mesh:
        print("\n---")
        print("Showing interactive 3D view of the main surface (level 0.0)...")
        print("(Close the window to exit the script)")
        # This will open a new window
        main_surface_mesh.show()
    else:
        print("Error: Could not extract main surface (level 0.0) to show.")

    print("---")
    print("Visualization complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not os.path.exists(filename):
            print(f"Error: File not found at {filename}")
            sys.exit(1)
        if not filename.endswith('.npy'):
            print(f"Error: File must be a .npy file")
            sys.exit(1)
    else:
        # Provide a default path for testing if no argument is given
        default_file = 'processed_meshes/03513137_1d1cc96025786db595f577622f465c85_1.npy'
        if os.path.exists(default_file):
             print(f"Warning: No file provided. Using default test file: {default_file}")
             filename = default_file
        else:
            print("Usage: python visualize_interactive.py <path_to_your_file.npy>")
            sys.exit(1)

    visualize_interactive(filename)