import os
import argparse
import torch
import numpy as np
import glob
import shutil
from tqdm import tqdm

# Import your model architecture
from src.model import MLPWithVoxel, NoiseScheduler

# ---- Configuration ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "exps/model.pth"
STATS_PATH = "exps/normalization_stats.npz"
VOXEL_DIR = "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_meshes"
# IMPORTANT: Path to your raw data to find the mesh.obj files
RAW_DATA_ROOT = "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1"

# Model Hyperparameters (MUST MATCH TRAINING)
MODEL_CONFIG = {
    "hidden_size": 128,
    "hidden_layers": 3,
    "emb_size": 128,
    "voxel_input_shape": (32, 32, 32),
    "num_timesteps": 1000
}

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = MLPWithVoxel(
        hidden_size=MODEL_CONFIG["hidden_size"],
        hidden_layers=MODEL_CONFIG["hidden_layers"],
        emb_size=MODEL_CONFIG["emb_size"],
        voxel_input_shape=MODEL_CONFIG["voxel_input_shape"]
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    scheduler = NoiseScheduler(
        num_timesteps=MODEL_CONFIG["num_timesteps"],
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )
    return model, scheduler

def get_random_test_voxel():
    files = glob.glob(os.path.join(VOXEL_DIR, "*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {VOXEL_DIR}")
    
    random_file = np.random.choice(files)
    filename = os.path.basename(random_file)
    print(f"Test Object Selected: {filename}")
    
    voxel_np = np.load(random_file).astype(np.float32)
    voxel_tensor = torch.from_numpy(voxel_np).float().unsqueeze(0)
    
    return voxel_tensor.to(device), filename

def find_original_mesh_path(npy_filename):
    """
    Reconstructs the path to mesh.obj from the processed filename.
    Format: <category>_<instance>_<id>.npy
    """
    name_parts = npy_filename.replace(".npy", "").split("_")
    # Last part is ID, first part is Category. Middle is instance (might contain underscores)
    category = name_parts[0]
    idx = name_parts[-1]
    instance = "_".join(name_parts[1:-1])
    
    mesh_path = os.path.join(RAW_DATA_ROOT, category, instance, idx, "mesh.obj")
    if not os.path.exists(mesh_path):
        # Try fallback if ID has leading zeros or slight mismatches
        print(f"Warning: Could not find mesh at {mesh_path}")
        return None
    return mesh_path

def denormalize_grasps(grasp_tensor, stats_path):
    stats = np.load(stats_path)
    feature_min = torch.from_numpy(stats["min"]).to(device)
    feature_max = torch.from_numpy(stats["max"]).to(device)
    grasp_real = (grasp_tensor + 1) / 2 * (feature_max - feature_min) + feature_min
    return grasp_real

def generate(model, scheduler, voxel_grid, num_samples=10):
    # print(f"Generating {num_samples} grasps...")
    voxel_batch = voxel_grid.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
    samples = torch.randn((num_samples, 19), device=device)
    
    with torch.no_grad():
        # Iterate without tqdm to keep log output clean during loop
        for t in reversed(range(scheduler.num_timesteps)):
            # FIX 1: Model needs a BATCH of timesteps
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            predicted_noise = model(samples, t_tensor, voxel_batch)
            
            # FIX 2: Scheduler step needs a SCALAR integer 't' to avoid "Boolean value of Tensor" error
            samples = scheduler.step(predicted_noise, t, samples)
            
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_objects", type=int, default=5, help="Number of random objects to process")
    parser.add_argument("--samples", type=int, default=10, help="Number of grasps per object")
    args = parser.parse_args()

    model, scheduler = load_model()
    
    print(f"\nStarting generation loop for {args.num_objects} objects...")

    for i in range(args.num_objects):
        print(f"\n--- Object {i+1}/{args.num_objects} ---")
        
        try:
            # 1. Get Data
            voxel_tensor, filename = get_random_test_voxel()
            
            # 2. Generate
            generated_normalized = generate(model, scheduler, voxel_tensor, num_samples=args.samples)
            generated_real = denormalize_grasps(generated_normalized, STATS_PATH)
            
            np_results = generated_real.cpu().numpy()
            
            # 3. Package Results
            base_name = filename.replace('.npy', '')
            output_dir = os.path.join("results", base_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save Grasps
            save_path = os.path.join(output_dir, "grasps.npz")
            fake_scores = np.ones(len(np_results)) 
            np.savez(save_path, grasps=np_results, scores=fake_scores)
            print(f"Saved {len(np_results)} grasps to {save_path}")
            
            # Copy Mesh
            original_mesh = find_original_mesh_path(filename)
            if original_mesh:
                shutil.copy(original_mesh, os.path.join(output_dir, "mesh.obj"))
                print(f"Copied mesh.obj to {output_dir}")
            else:
                print("Warning: Could not find original mesh.obj to copy.")
                
        except Exception as e:
            print(f"Error processing object: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nGeneration loop complete. Check 'results/' folder.")