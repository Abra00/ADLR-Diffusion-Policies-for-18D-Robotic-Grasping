import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from src.Diff_model import MLPWithVoxel, NoiseScheduler
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameters ----
config = {
    "batch_size": 256,              # Training batch size
    "eval_batch_size": 1000,        # Batch size for evaluation
    "epochs": 100,                  # Total training epochs
    "learning_rate": 1e-3,          # Optimizer learning rate
    "num_timesteps": 50,            # Number of diffusion timesteps
    "hidden_size": 512,             # Hidden layer size in the MLP
    "hidden_layers": 4,             # Number of hidden layers
    "embedding_size": 128,          # Embedding size for voxel input
    "save_images_step": 50          # How often to generate sample grasps
}
wandb.init(
    project="adlr-diffusion_grasping",
    job_type="training_diff_test",
    config=config 
)
# ---- Load voxel and grasp data ----
voxel_dir = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_meshes"
grasp_dir = grasp_dir = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_scores"

objects = []

# List all voxel files
voxel_files = sorted([f for f in os.listdir(voxel_dir) if f.endswith(".npy")])

for vf in voxel_files:
    voxel_path = os.path.join(voxel_dir, vf)
    grasp_path = os.path.join(grasp_dir, vf.replace(".npy", ".npz"))
    
    if not os.path.exists(grasp_path):
        print(f"Warning: No grasp found for {vf}, skipping.")
        continue

    # Load voxel grid and add channel dimension
    voxel = np.load(voxel_path).astype(np.float32)        # Shape: (32,32,32)
    voxel = voxel.reshape(1, 32, 32, 32)                  # Shape: (1,32,32,32)

    # Load grasp data
    grasp_data = np.load(grasp_path)
    grasps = grasp_data["grasps"].astype(np.float32)      # Shape: (N,19)

    objects.append({
        "voxel_grid": voxel,
        "grasps": grasps,
        "object_id": vf.replace(".npy","")
    })

print("Loaded objects:", len(objects))

# Concatenate all grasps to compute per-feature min/max for normalization
all_grasps = np.concatenate([obj["grasps"] for obj in objects], axis=0)  # Shape: (N,19)
feature_min = all_grasps.min(axis=0)  # Minimum for each grasp dimension
feature_max = all_grasps.max(axis=0)  # Maximum for each grasp dimension

# ---- Normalize grasps to [-1, 1] per feature ----
for obj in objects:
    obj["grasps"] = 2 * (obj["grasps"] - feature_min) / (feature_max - feature_min + 1e-6) - 1

# ---- Dataset Class ----
class GraspDataset(Dataset):
    def __init__(self, objects):
        self.items = []
        for obj in objects:
            voxel = obj["voxel_grid"]        # Voxel grid for this object
            grasps = obj["grasps"]           # Normalized grasps
            obj_id = obj["object_id"]
            for g in grasps:
                # Store a single grasp along with its voxel and object ID
                self.items.append({
                    "voxel": voxel,
                    "grasp": g,
                    "object_id": obj_id
                })
                break  # Only take first grasp for simplicity
            break  # Only take first object for now

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Return grasp tensor, voxel tensor, and object ID
        return (torch.from_numpy(item["grasp"]).float(),
                torch.from_numpy(item["voxel"]).float(),
                item["object_id"])

# ---- Prepare dataset and dataloader ----
train_dataset = GraspDataset(objects)

# Stack all voxels into a single tensor for reference
all_voxels = torch.tensor(np.concatenate([obj["voxel_grid"][None] for obj in objects], axis=0))

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

# ---- Model, optimizer, and learning rate scheduler ----
model = MLPWithVoxel(
    hidden_size=config["hidden_size"],
    hidden_layers=config["hidden_layers"],
    emb_size=config["embedding_size"],
    voxel_input_shape=(32,32,32)
).to(device)

# Noise scheduler handles forward and reverse diffusion
noise_scheduler = NoiseScheduler(num_timesteps=config["num_timesteps"])

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
"""scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(              # no scheduler for overfitting 
    optimizer,
    T_max=config["epochs"],
    eta_min=1e-6
)"""

# ---- Training loop ----
losses = []
generated_data = []

for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{config["epochs"]}]')
    
    for i, (x_batch, voxel_batch, obj_id_batch) in enumerate(train_loader):
        B = x_batch.shape[0]
        
        # If batch is smaller than desired, repeat samples to fill batch
        if B < config["batch_size"]:
            repeat_factor = (config["batch_size"] + B - 1) // B
            x_batch = x_batch.repeat(repeat_factor, 1)[:config["batch_size"]]
            voxel_batch = voxel_batch.repeat(repeat_factor, 1, 1, 1, 1)[:config["batch_size"]]
        
        x_batch = x_batch.to(device)          # Grasp vectors (B,19)
        voxel_batch = voxel_batch.to(device)  # Voxel grids (B,1,32,32,32)
        
        # Add random Gaussian noise to grasps
        noise = torch.randn_like(x_batch)
        # Randomly select timesteps for forward diffusion
        timesteps = torch.randint(0, config["num_timesteps"], (x_batch.shape[0],), device=device)
        # Apply forward diffusion to add noise
        noisy = noise_scheduler.add_noise(x_batch, noise, timesteps)
        
        # Forward pass: model predicts the noise
        noise_pred = model(noisy, timesteps, voxel_batch)
        
        # Compute MSE loss between predicted and true noise
        loss = nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        # Log step loss
        wandb.log({"train/loss": loss.item(), "epoch": epoch})
        epoch_loss += loss.item()
        
        # Update progress bar with current average loss
        train_loader_tqdm.set_postfix(curr_train_loss=f"{epoch_loss / (i + 1):.8f}")
    wandb.log({"epoch_loss": epoch_loss / len(train_loader), "epoch": epoch})
    
    # Step learning rate scheduler at the end of the epoch
    #scheduler.step().  # not needed for overfitting 
    print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss / len(train_loader):.6f}")

    # ---- Generate sample grasps ----
    if epoch % config["save_images_step"] == 0 or epoch == config["epochs"]-1:
        model.eval()
        num_samples_to_generate = min(5, len(train_dataset))

        for i in range(num_samples_to_generate):
            grasp_tensor, voxel, obj_id = train_dataset[i]
            voxel = voxel.unsqueeze(0).to(device)  # Add batch dimension

            # Start reverse diffusion from pure noise
            sample = torch.randn(1, 19, device=device)

            # Reverse diffusion loop
            with torch.no_grad():
                for t in reversed(range(config["num_timesteps"])):
                    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                    residual = model(sample, t_tensor, voxel)
                    sample = noise_scheduler.step(residual, t_tensor, sample)

            # Denormalize generated grasp back to original scale
            grasp_np_norm = sample.cpu().numpy()[0]
            grasp_np = (grasp_np_norm + 1)/2 * (feature_max - feature_min) + feature_min

            # Store generated grasps with object ID
            generated_data.append((obj_id, grasp_np))
wandb.finish()
# ---- Save results ----
generated_data = np.array(generated_data, dtype=object)
os.makedirs("exps", exist_ok=True)
torch.save(model.state_dict(), "exps/model.pth")
np.save("exps/generated_grasps_with_id.npy", np.array(generated_data, dtype=object), allow_pickle=True)
np.save("exps/loss.npy", losses)

