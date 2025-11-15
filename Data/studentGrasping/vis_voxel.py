import numpy as np
import matplotlib.pyplot as plt
import trimesh

# -------------------------------
# Settings
# -------------------------------
file_path = "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/models/test_outputs/reconstructed_1.npy"
slice_axis = 2  # axes for 2D Slice (0=x, 1=y, 2=z)
threshold = 0.0  # everything  <= threshold is counted as inside the mesh 

# -------------------------------
# 1. Load SDF
# -------------------------------
sdf = np.load(file_path)
print(f"SDF shape: {sdf.shape}, min: {sdf.min()}, max: {sdf.max()}")

# -------------------------------
# 2. Visualize middle 2D slice
# -------------------------------
slice_idx = sdf.shape[slice_axis] // 2
if slice_axis == 0:
    slice_2d = sdf[slice_idx, :, :]
elif slice_axis == 1:
    slice_2d = sdf[:, slice_idx, :]
else:
    slice_2d = sdf[:, :, slice_idx]

plt.figure(figsize=(5,5))
plt.imshow(slice_2d, cmap="seismic", origin="lower")
plt.colorbar(label="Signed Distance")
plt.title(f"Middle slice along axis {slice_axis}")
plt.show()

# -------------------------------
# 3. Convert SDF to boolean voxel grid
# -------------------------------
voxels = sdf <= threshold  # True = inside mesh

# -------------------------------
# 4. Create VoxelGrid and render
# -------------------------------
voxel_grid = trimesh.voxel.VoxelGrid(voxels)
voxel_mesh = voxel_grid.as_boxes()
voxel_mesh.show()
