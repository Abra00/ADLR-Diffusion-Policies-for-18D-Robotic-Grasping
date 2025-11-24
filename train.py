import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb

# Import custom modules
from src.model import MLPWithVoxel, NoiseScheduler
from src.data_loader import load_and_process_data

# ---- Configuration ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Paths
VOXEL_DIR = "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_meshes"
GRASP_DIR = "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_scores"

config = {
    "batch_size": 128,              
    "eval_batch_size": 1000,        
    "epochs": 20,                   
    "learning_rate": 2e-4,          
    "num_timesteps": 1000,           
    "hidden_size": 128,             
    "hidden_layers": 3,             
    "embedding_size": 128,          
    "save_images_step": 50          
}

# Initialize WandB
wandb.init(
    project="adlr-diffusion_grasping",
    job_type="training_diff",
    config=config 
)

# ---- 1. Load Data ----
print("Loading Data...")
# We get feature_min and feature_max here. These are CRITICAL for using the model later.
train_loader, val_loader, train_dataset, val_dataset, feature_min, feature_max = load_and_process_data(
    voxel_dir=VOXEL_DIR, 
    grasp_dir=GRASP_DIR, 
    batch_size=config["batch_size"]
)

# ---- 2. Initialize Model ----
model = MLPWithVoxel(
    hidden_size=config["hidden_size"],
    hidden_layers=config["hidden_layers"],
    emb_size=config["embedding_size"],
    voxel_input_shape=(32,32,32)
).to(device)

noise_scheduler = NoiseScheduler(num_timesteps=config["num_timesteps"], device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=1e-6
)

# ---- 3. Training Loop ----
losses = []
generated_data = []

for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{config["epochs"]}]')
    
    for i, (x_batch, voxel_batch, obj_id_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)          
        voxel_batch = voxel_batch.to(device)  
        
        # Add random Gaussian noise
        noise = torch.randn_like(x_batch)
        timesteps = torch.randint(0, config["num_timesteps"], (x_batch.shape[0],), device=device)
        noisy = noise_scheduler.add_noise(x_batch, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy, timesteps, voxel_batch)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        wandb.log({"train/loss": loss.item(), "epoch": epoch})
        epoch_loss += loss.item()
        train_loader_tqdm.set_postfix(curr_train_loss=f"{epoch_loss / (i + 1):.8f}")
    
    avg_train_loss = epoch_loss / len(train_loader)
    wandb.log({"epoch_loss": avg_train_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})
    scheduler.step()
    print(f"Epoch {epoch+1} finished | Avg Loss: {avg_train_loss:.6f}")

    # ---- Validation ----
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for i, (x_val, voxel_val, _) in enumerate(val_loader):
            x_val = x_val.to(device)
            voxel_val = voxel_val.to(device)

            noise = torch.randn_like(x_val)
            timesteps = torch.randint(0, config["num_timesteps"], (x_val.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(x_val, noise, timesteps)
            
            noise_pred = model(noisy, timesteps, voxel_val)
            val_loss = nn.functional.mse_loss(noise_pred, noise)
            val_losses.append(val_loss.item())
    
    avg_val_loss = np.mean(val_losses)
    wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

    # ---- Generate example grasps (Sanity Check) ----
    if epoch % config["save_images_step"] == 0 or epoch == config["epochs"]-1:
        model.eval()
        num_samples_to_generate = min(5, len(val_dataset))

        for i in range(num_samples_to_generate):
            _, voxel, obj_id = val_dataset[i]
            voxel = voxel.unsqueeze(0).to(device) 

            sample = torch.randn(1, 19, device=device)

            with torch.no_grad():
                for t in reversed(range(config["num_timesteps"])):
                    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                    residual = model(sample, t_tensor, voxel)
                    sample = noise_scheduler.step(residual, t_tensor, sample)

            # Denormalize just for this sanity check
            grasp_np_norm = sample.cpu().numpy()[0]
            grasp_np = (grasp_np_norm + 1)/2 * (feature_max - feature_min) + feature_min

            generated_data.append((obj_id, grasp_np))

wandb.finish()

# ---- Save model and NORMALIZATION STATS ----
os.makedirs("exps", exist_ok=True)

# 1. Save Model Weights
torch.save(model.state_dict(), "exps/model.pth")

# 2. Save Normalization Statistics (CRITICAL)
# We save these so we can "un-normalize" new predictions later
np.savez("exps/normalization_stats.npz", min=feature_min, max=feature_max)

# 3. Save other logs
generated_data = np.array(generated_data, dtype=object)
np.save("exps/generated_grasps_with_id.npy", generated_data, allow_pickle=True)
np.save("exps/loss.npy", losses)

print("Training complete.")
print("Model saved to exps/model.pth")
print("Normalization stats saved to exps/normalization_stats.npz")



# #For inference
# # Load stats
# stats = np.load("exps/normalization_stats.npz")
# feature_min = stats["min"]
# feature_max = stats["max"]

# # ... Generate a grasp using the model ...
# prediction_normalized = model(...) 

# # Denormalize
# prediction_real = (prediction_normalized + 1)/2 * (feature_max - feature_min) + feature_min