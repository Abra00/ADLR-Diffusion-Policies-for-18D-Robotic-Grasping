import torch
import numpy as np
import argparse
import pybullet
import pybullet_data
import os
import time
from src.model import MLPWithVoxel, NoiseScheduler

# ---- CONFIGURATION ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Update these paths if necessary
MODEL_PATH = "exps/model_final.pth"
STATS_PATH = "exps/normalization_stats.npz"

MODEL_CONFIG = {
    "hidden_size": 256, 
    "hidden_layers": 4, 
    "emb_size": 128, 
    "voxel_input_shape": (32, 32, 32) 
}

def generate_grasp(model, scheduler, voxel_grid):
    """ Generates a grasp from pure noise using Reverse Diffusion """
    # 1. Start with Gaussian Noise
    sample = torch.randn(1, 19, device=DEVICE)
    
    # 2. Prepare Voxel Grid (Batch=1, Channel=1, D, H, W)
    if voxel_grid.ndim == 3:
        voxel_grid = voxel_grid[None, None, ...]
    elif voxel_grid.ndim == 4:
        voxel_grid = voxel_grid[None, ...]
    voxel_grid = torch.from_numpy(voxel_grid).float().to(DEVICE)
    
    # 3. Denoising Loop
    with torch.no_grad():
        for t in reversed(range(1000)):
            t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
            res = model(sample, t_tensor, voxel_grid)
            sample = scheduler.step(res, t_tensor, sample)
            
    return sample.cpu().numpy().flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to processed .npz file (e.g. processed_data/object_id.npz)")
    parser.add_argument("obj_path", help="Path to RAW .obj file (e.g. student_grasps_v1/.../mesh.obj)")
    parser.add_argument("--urdf", default="/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/urdfs/dlr2.urdf", help="Path to robot URDF")
    parser.add_argument("--ground_truth", action="store_true", help="Visualize the saved ground truth instead of model prediction")
    args = parser.parse_args()

    # --- 1. Load Stats & Metadata ---
    if not os.path.exists(STATS_PATH):
        print(f"Error: Stats file not found at {STATS_PATH}")
        return
    
    stats = np.load(STATS_PATH)
    j_min = stats["min"][7:]
    j_max = stats["max"][7:]

    if not os.path.exists(args.npz_path):
        print(f"Error: NPZ file not found at {args.npz_path}")
        return
        
    data = np.load(args.npz_path)
    voxel = data["voxel_sdf"]
    center = data["center"] # Used to un-center the grasp
    scale = data["scale"]   # Used to un-scale the grasp

    # --- 2. Get the Grasp Vector ---
    if args.ground_truth:
        print("\n--- MODE: GROUND TRUTH ---")
        print("Visualizing the raw data from the .npz file (reversing normalization).")
        # In your processed file, 'grasps' are: 
        #   Pos: Centered & Scaled
        #   Rot: Raw Quaternion
        #   Joints: Raw (0.0 - 1.5 rads)
        
        # Pick the best grasp (assuming sorted or taking first valid)
        raw_vec = data["grasps"][0]
        
        # Denormalize Position Only
        pos = (raw_vec[0:3] / scale) + center
        
        # Rot and Joints are already raw in your .npz (based on process_data_v2.py)
        rot = raw_vec[3:7]
        joints = raw_vec[7:]
        
    else:
        print("\n--- MODE: MODEL INFERENCE ---")
        print("Generating new grasp from noise...")
        
        # Load Model
        model = MLPWithVoxel(**MODEL_CONFIG).to(DEVICE)
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            return
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        scheduler = NoiseScheduler(num_timesteps=1000, device=DEVICE)

        # Generate
        norm_grasp = generate_grasp(model, scheduler, voxel)

        # Denormalize EVERYTHING
        # 1. Pos: Un-scale, then un-center
        pos = (norm_grasp[0:3] / scale) + center
        
        # 2. Rot: Normalize quaternion
        rot = norm_grasp[3:7]
        rot = rot / np.linalg.norm(rot)
        
        # 3. Joints: Un-map [-1, 1] -> [Min, Max]
        joints = (norm_grasp[7:] + 1) / 2 * (j_max - j_min) + j_min

    print(f"\nVisualizing Grasp at Position: {pos}")
    
    # --- 3. PyBullet Visualization ---
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    
    # A. Load Object (Raw Mesh) at (0,0,0) World
    # This matches the original script logic
    visualShapeId = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=args.obj_path,
        meshScale=[1, 1, 1],
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.4, 0.4, 0]
    )
    pybullet.createMultiBody(
        baseVisualShapeIndex=visualShapeId,
        basePosition=[0,0,0],
        baseOrientation=[0,0,0,1]
    )

    # B. Load Hand with CRITICAL FLAG
    hand_id = pybullet.loadURDF(
        args.urdf,
        basePosition=pos,
        baseOrientation=rot,
        useFixedBase=True,
        flags=pybullet.URDF_MAINTAIN_LINK_ORDER # <--- THE FIX
    )

    # C. Set Joints
    # Using the exact indices from the original script
    active_joints = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]
    coupled_joints = [3, 9, 15, 21] # (active_joint_idx -> coupled_joint_idx is +1)

    for k, j_idx in enumerate(active_joints):
        if k < len(joints):
            val = joints[k]
            # Set Active
            pybullet.resetJointState(hand_id, j_idx, val)
            
            # Set Coupled (if applicable)
            if j_idx in coupled_joints:
                pybullet.resetJointState(hand_id, j_idx + 1, val)

    print("\nSimulation running. Press Enter in terminal to close.")
    input()
    pybullet.disconnect()

if __name__ == "__main__":
    main()