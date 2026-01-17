# run with: python -m Testing.generate_grasps_testset
########
# create Grasps 
#######
import torch
import numpy as np
import argparse
import pybullet
import pybullet_data
import os
from pathlib import Path
from src.model import DiffusionMLP
from src.noise_scheduler import NoiseScheduler
import random

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

def generate_grasps(model, scheduler, voxel_grid, num_samples=3):
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
    #########################
    # create Grasps         #
    #########################
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=False, help="Path to .pth file", default= "exps_new/model_final.pth")
    parser.add_argument("--npz_path", required=False, help="Path to processed .npz file", default=".\Data\Testset\Processed_Data_MultiGripperGrasp")
    parser.add_argument("--obj_path", required=False, help="Path to raw .obj mesh", default=".C:\Data\Testset\MultiGrippperGrasp")
    parser.add_argument("--urdf_path", default="./Data/studentGrasping/urdfs/dlr2.urdf", help="Path to robot URDF")
    parser.add_argument("--num_grasps", type=int, default=3, help="How many to generate")
    args = parser.parse_args()

    print("Loading Model...")
    model = DiffusionMLP(
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["hidden_layers"],
        emb_size=model_config['emb_size'],
        input_emb_dim=model_config['input_emb_dim'],
        scale=model_config['scale']
    )

    PROCESSED_DIR = Path(args.npz_path)  
    npz_files = list(PROCESSED_DIR.glob("*.npz"))
    npz_files = npz_files[:10] #for testing
    print(f"Found {len(npz_files)} processed voxel files")

    for npz_path in npz_files:
        print(f"Processing {npz_path.name} ...")
        data = np.load(npz_path)
        voxel = data["voxel_sdf"]
        center = data["center"]
        scale = data["scale"]


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
        base_name = npz_path.stem
        output_dir = Path("Testing/generated_grasps")     #create new folder if neccesary
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / f"{base_name}_generated_grasps.npz"
        np.savez(save_path,
                position=pos_array,
                orientation=rot_array,
                joints=joints_array)

        print(f"Saved generated grasps to {save_path}")
    #show generated grasps for debugging
    num_to_show = 10
    GENERATED_DIR = Path("Testing/generated_grasps")
    npz_files_to_show = random.sample(
        list(GENERATED_DIR.glob("*.npz")), 
        min(num_to_show, len(list(GENERATED_DIR.glob("*.npz"))))
    )
    print(f"Visualizing {len(npz_files_to_show)} random files (first grasp only).")

    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    active_joints = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]
    coupled_joints = [3, 9, 15, 21]

    for npz_path in npz_files_to_show:
        print(f"Visualizing {npz_path.name} ...")

        # Loade voxel
        data = np.load(npz_path)
        pos = data['position'][0]      #first grasp 
        rot = data['orientation'][0]
        joints = data['joints'][0]

        # get mesh path
        base_name = npz_path.stem.replace("_generated_grasps", "")
        obj_path = Path("Data/Testset/MultiGrippperGrasp/GoogleScannedObjects") / base_name / "meshes/model_centered.obj"
        if not obj_path.exists():
            print(f"Mesh not found: {obj_path}")

        # delete old objects
        pybullet.resetSimulation()
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())


        # Load object 
        visualShapeId = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=str(obj_path),
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            meshScale=[1, 1, 1]
        )
        pybullet.createMultiBody(
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )

        # Load hand 
        hand_id = pybullet.loadURDF(
            args.urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=pybullet.URDF_MAINTAIN_LINK_ORDER
        )

        # # Set hand to first grasp
        pybullet.resetBasePositionAndOrientation(hand_id, pos, rot)
        for k, j_idx in enumerate(active_joints):
            val = joints[k]
            pybullet.resetJointState(hand_id, j_idx, val)
            if j_idx in coupled_joints:
                pybullet.resetJointState(hand_id, j_idx + 1, val)

        input("Press ENTER to see the next grasp...")

    pybullet.disconnect()
    #########################
    # simulate              #
    #########################



if __name__ == "__main__":
    main()

#####
# run simulation 
####


####
# safe results 
####