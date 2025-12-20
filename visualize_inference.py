import torch
import numpy as np
import argparse
import pybullet
import pybullet_data
import os
from pathlib import Path
from src.model import DiffusionMLP
from src.noise_scheduler import NoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Update this to match your saved model
model_config = {
    "hidden_size": 512,
    "hidden_layers": 6, 
    "emb_size": 256, 
    "input_emb_dim": 64,
    "scale": 2500
}

# Physical Limits (from urdf)
finger_min = [-0.523598, -0.349065, -0.174532]
finger_max = [ 0.523598,  1.500983,  1.832595]
joints_min = np.array(finger_min * 4, dtype=np.float32)
joints_max = np.array(finger_max * 4, dtype=np.float32)

def generate_grasps(model, scheduler, voxel_grid, num_samples=10):
    """ 
    Generates 'num_samples' grasps in parallel.
    """
    model.eval()
    
    voxel_batch = torch.from_numpy(voxel_grid).float().to(device)
    if voxel_batch.ndim == 3:
        voxel_batch = voxel_batch.unsqueeze(0).unsqueeze(0) # (1, 1, 32, 32, 32)
    elif voxel_batch.ndim == 4:
        voxel_batch = voxel_batch.unsqueeze(0) # (1, 1, 32, 32, 32)
        
    voxel_batch = voxel_batch.repeat(num_samples, 1, 1, 1, 1)

    sample = torch.randn(num_samples, 19, device=device)
    
    # 3. Denoising Loop
    print(f"Generating {num_samples} grasps...")
    with torch.no_grad():
        for t in reversed(range(scheduler.num_timesteps)):
            # Create batch of timesteps (e.g. [999, 999, ...])
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(sample, t_batch, voxel_batch)
            
            # Remove noise (Reverse Step)
            # You need to implement 'step' in your scheduler, 
            # OR use this simplified logic if you haven't implemented it yet:
            sample = scheduler.step(pred_noise, t, sample)
            
    return sample.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=False, help="Path to .pth file", default= "exps_new/model_final.pth")
    parser.add_argument("--npz_path", required=False, help="Path to processed .npz file", default="./Data/studentGrasping/processed_data_new/02747177_1c3cf618a6790f1021c6005997c63924_0.npz")
    parser.add_argument("--obj_path", required=False, help="Path to raw .obj mesh", default="./Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/mesh.obj")
    parser.add_argument("--urdf_path", default="./Data/studentGrasping/urdfs/dlr2.urdf", help="Path to robot URDF")
    parser.add_argument("--num_grasps", type=int, default=10, help="How many to generate")
    args = parser.parse_args()

    print(f"Loading data from {args.npz_path}...")
    data = np.load(args.npz_path)
    voxel = data["voxel_sdf"]
    center = data["center"]
    scale = data["scale"]

    print("Loading Model...")
    model = DiffusionMLP(
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["hidden_layers"],
        emb_size=model_config['emb_size'],
        input_emb_dim=model_config['input_emb_dim'],
        scale=model_config['scale']
    )
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    scheduler = NoiseScheduler(num_timesteps=1000, device=device)

    norm_grasps = generate_grasps(model, scheduler, voxel, args.num_grasps)

    real_grasps = []
    for g in norm_grasps:
        # Position: Un-scale, then un-center
        pos = (g[0:3] / scale) + center
        
        # Rotation: Normalize quaternion
        rot = g[3:7]
        rot = rot / np.linalg.norm(rot)
        
        # Joints: Un-map [-1, 1] -> [Min, Max]
        joints = (g[7:] + 1) / 2 * (joints_max - joints_min) + joints_min
        
        real_grasps.append((pos, rot, joints))
    #safe grasps
    pos_array = np.array([g[0] for g in real_grasps])
    rot_array = np.array([g[1] for g in real_grasps])
    joints_array = np.array([g[2] for g in real_grasps])
    base_name = Path(args.npz_path).stem
    output_dir = Path("generated_grasps")     #create new folder if neccesary
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"{base_name}_generated_grasps.npz"
    np.savez(save_path,
            position=pos_array,
            orientation=rot_array,
            joints=joints_array)

    print(f"Saved generated grasps to {save_path}")
    #simulate 
    print("Starting Simulation...")
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load Object
    visualShapeId = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=args.obj_path,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.4, 0.4, 0],
        meshScale=[1, 1, 1]
    )
    pybullet.createMultiBody(
        baseVisualShapeIndex=visualShapeId,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1]
    )

    hand_id = pybullet.loadURDF(
        args.urdf_path,
        basePosition=[0,0,0],
        useFixedBase=True,
        flags=pybullet.URDF_MAINTAIN_LINK_ORDER
    )

    active_joints = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]
    coupled_joints = [3, 9, 15, 21] 

    print(f"\nVisualizing {len(real_grasps)} generated grasps.")
    print("Press ENTER in the terminal to see the next grasp...")

    for i, (pos, rot, joints) in enumerate(real_grasps):
        print(f"Grasp {i+1}/{len(real_grasps)}")
        
        # Move Hand Base
        pybullet.resetBasePositionAndOrientation(hand_id, pos, rot)
        
        # Move Hand Fingers
        for k, j_idx in enumerate(active_joints):
            val = joints[k]
            pybullet.resetJointState(hand_id, j_idx, val)
            
            # Handle Coupled Joints
            if j_idx in coupled_joints:
                pybullet.resetJointState(hand_id, j_idx + 1, val)
        
        input()

    pybullet.disconnect()

if __name__ == "__main__":
    main()