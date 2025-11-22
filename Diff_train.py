voxel_dir = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_meshes"
grasp_dir = grasp_dir = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_scores"
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
    "batch_size": 128,              # Training batch size
    "eval_batch_size": 1000,        # Batch size for validation
    "epochs": 20,                   # Number of training epochs
    "learning_rate": 1e-3,          # Optimizer learning rate
    "num_timesteps": 100,           # Number of diffusion timesteps
    "hidden_size": 128,             # Hidden layer size in the MLP
    "hidden_layers": 3,             # Number of hidden layers
    "embedding_size": 128,          # Size of embedding vector for voxel input
    "save_images_step": 50          # How often to generate and save samples
}
wandb.init(
    project="adlr-diffusion_grasping",
    job_type="training_diff",
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

# Concatenate all grasps to compute min/max for normalization
all_grasps = np.concatenate([obj["grasps"] for obj in objects], axis=0)
feature_min = all_grasps.min(axis=0)  # Per-feature minimum
feature_max = all_grasps.max(axis=0)  # Per-feature maximum

# ---- Normalize grasps to [-1, 1] ----
for obj in objects:
    obj["grasps"] = 2 * (obj["grasps"] - feature_min) / (feature_max - feature_min + 1e-6) - 1

# ---- Dataset Class ----
class GraspDataset(Dataset):
    def __init__(self, objects):
        self.items = []
        for obj in objects:
            voxel = obj["voxel_grid"]        # Voxel grid for the object
            grasps = obj["grasps"]           # Normalized grasps
            obj_id = obj["object_id"]
            for g in grasps:
                # Store each grasp with its corresponding voxel and object ID
                self.items.append({
                    "voxel": voxel,
                    "grasp": g,
                    "object_id": obj_id
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Return grasp tensor, voxel tensor, and object ID
        return (torch.from_numpy(item["grasp"]).float(),
                torch.from_numpy(item["voxel"]).float(),
                item["object_id"])

# ---- Split data into train and validation sets ----
train_objs, val_objs = train_test_split(objects, test_size=0.2)
train_dataset = GraspDataset(train_objs)
val_dataset = GraspDataset(val_objs)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# ---- Model, optimizer, and scheduler ----
model = MLPWithVoxel(
    hidden_size=config["hidden_size"],
    hidden_layers=config["hidden_layers"],
    emb_size=config["embedding_size"],
    voxel_input_shape=(32,32,32)
).to(device)

noise_scheduler = NoiseScheduler(num_timesteps=config["num_timesteps"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=1e-6
)

# ---- Training loop ----
losses = []
generated_data = []

for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{config["epochs"]}]')
    
    for i, (x_batch, voxel_batch, obj_id_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)          # Grasp vectors (B,19)
        voxel_batch = voxel_batch.to(device)  # Voxel grids (B,1,32,32,32)
        
        # Add random Gaussian noise to the grasps
        noise = torch.randn_like(x_batch)
        # Sample random timesteps for diffusion
        timesteps = torch.randint(0, config["num_timesteps"], (x_batch.shape[0],), device=device)
        # Apply forward diffusion (noisy version of x_batch)
        noisy = noise_scheduler.add_noise(x_batch, noise, timesteps)
        
        # Forward pass: predict noise
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
        train_loader_tqdm.set_postfix(curr_train_loss=f"{epoch_loss / (i + 1):.8f}")
    wandb.log({"epoch_loss": epoch_loss / len(train_loader), "lr": scheduler.get_last_lr()[0], "epoch": epoch})
    scheduler.step()
    print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss / len(train_loader):.6f}")

    # ---- Validation ----
    model.eval()
    val_losses = []
    validation_loss = 0
    validation_loader_tqdm = tqdm(val_loader, desc=f'Validation Epoch [{epoch + 1}/{config["epochs"]}]')
    with torch.no_grad():
        for i, (x_val, voxel_val, _) in enumerate(val_loader):
            x_val = x_val.to(device)
            voxel_val = voxel_val.to(device)

            # Add noise for validation data
            noise = torch.randn_like(x_val)
            timesteps = torch.randint(0, config["num_timesteps"], (x_val.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(x_val, noise, timesteps)
            
            noise_pred = model(noisy, timesteps, voxel_val)
            val_loss = nn.functional.mse_loss(noise_pred, noise)
            val_losses.append(val_loss.item())
            validation_loss += val_loss.item()
            validation_loader_tqdm.set_postfix(val_loss = "{:.8f}".format(validation_loss / (i + 1)))
    
    avg_val_loss = np.mean(val_losses)

    # ---- Generate example grasps ----
    if epoch % config["save_images_step"] == 0 or epoch == config["epochs"]-1:
        model.eval()
        num_samples_to_generate = min(5, len(val_dataset))

        for i in range(num_samples_to_generate):
            grasp_tensor, voxel, obj_id = val_dataset[i]
            voxel = voxel.unsqueeze(0).to(device)  # Add batch dimension

            # Start reverse diffusion from pure noise
            sample = torch.randn(1, 19, device=device)

            # Perform reverse diffusion to generate sample grasp
            with torch.no_grad():
                for t in reversed(range(config["num_timesteps"])):
                    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                    residual = model(sample, t_tensor, voxel)
                    sample = noise_scheduler.step(residual, t_tensor, sample)

            # Denormalize generated grasp
            grasp_np_norm = sample.cpu().numpy()[0]
            grasp_np = (grasp_np_norm + 1)/2 * (feature_max - feature_min) + feature_min

            # Store generated grasps with object IDs
            generated_data.append((obj_id, grasp_np))
wandb.finish()
# ---- Save model and results ----
generated_data = np.array(generated_data, dtype=object)
os.makedirs("exps", exist_ok=True)
torch.save(model.state_dict(), "exps/model.pth")
np.save("exps/generated_grasps_with_id.npy", np.array(generated_data, dtype=object), allow_pickle=True)
np.save("exps/loss.npy", losses)


