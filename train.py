import os
# Fix for OpenMP error on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb

# Import custom modules
from src.model import MLPWithVoxel, NoiseScheduler
from src.data_loader import load_and_process_data

# ---- Optimization Flags ----
# Enable CuDNN benchmark for speed boost on fixed input sizes
torch.backends.cudnn.benchmark = True

# ---- Configuration ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Updated Paths for the "Final" processed data
# Note: With the new script, both voxels and grasps are in the same .npz files in one folder.
# You pass the same folder to both arguments if your loader expects two, 
# or ensure load_and_process_data handles the single-folder structure.
DATA_ROOT = "/home/abra/Workspace/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_data_final"

config = {
    "batch_size": 64,
    "eval_batch_size": 1,
    "epochs": 2000,
    "learning_rate": 2e-4,
    "num_timesteps": 1000,
    "hidden_size": 128,
    "hidden_layers": 3,
    "embedding_size": 128,
    "save_images_step": 50,  # Visualize every 5 epochs
    "resolution": 32,        # Match the resolution from processing script
    "test_size": 100
}

# Initialize WandB
wandb.init(
    project="adlr-diffusion_grasping",
    job_type="training_diff",
    config=config 
)

# ---- 1. Load Data ----
print("Loading Data...")
# We use the same path for both because the new .npz files contain everything
train_loader, val_loader, train_dataset, val_dataset, feature_min, feature_max = load_and_process_data(
    data_dir=DATA_ROOT, 
    batch_size=config["batch_size"],
    test_size=config["test_size"]
)
print(f"--> TRAINING ON {len(train_dataset)} SAMPLES. (If this is > 100, you are not overfitting on a single item!)")

# ---- 2. SAVE STATS IMMEDIATELY (Safety) ----
os.makedirs("exps", exist_ok=True)
np.savez("exps/normalization_stats.npz", min=feature_min, max=feature_max)
print("Normalization stats saved to exps/normalization_stats.npz")

# ---- 3. Initialize Model & Optimizer ----
# Critical: Update input_shape to (64,64,64) to match your new data processing
model = MLPWithVoxel(
    hidden_size=config["hidden_size"],
    hidden_layers=config["hidden_layers"],
    emb_size=config["embedding_size"],
    voxel_input_shape=(config["resolution"], config["resolution"], config["resolution"])
).to(device)

noise_scheduler = NoiseScheduler(num_timesteps=config["num_timesteps"], device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=1e-6
)

# Initialize Scaler for Automatic Mixed Precision (AMP)
scaler = torch.amp.GradScaler('cuda')

# ---- 4. Training Loop ----
best_val_loss = float('inf')
losses = []
generated_data = []

for epoch in range(config["epochs"]):
    # --- TRAIN ---
    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{config["epochs"]}]')
    
    for i, (x_batch, voxel_batch, obj_id_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        voxel_batch = voxel_batch.to(device)
        
        # Generate noise and timestamps
        noise = torch.randn_like(x_batch)
        timesteps = torch.randint(0, config["num_timesteps"], (x_batch.shape[0],), device=device)
        noisy = noise_scheduler.add_noise(x_batch, noise, timesteps)
        
        optimizer.zero_grad() 
        
        # Run Forward Pass with AMP (Mixed Precision)
        with torch.amp.autocast('cuda'):
            noise_pred = model(noisy, timesteps, voxel_batch)
            loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        losses.append(loss.item())
        wandb.log({"train/loss": loss.item(), "epoch": epoch})
        epoch_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=f"{loss.item():.6f}")
    
    avg_train_loss = epoch_loss / len(train_loader)
    wandb.log({"epoch_loss": avg_train_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})
    scheduler.step()
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}")

    # --- VALIDATION ---
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for i, (x_val, voxel_val, _) in enumerate(val_loader):
            x_val = x_val.to(device)
            voxel_val = voxel_val.to(device)

            noise = torch.randn_like(x_val)
            timesteps = torch.randint(0, config["num_timesteps"], (x_val.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(x_val, noise, timesteps)
            
            # Use AMP for validation too (faster)
            with torch.amp.autocast('cuda'):
                noise_pred = model(noisy, timesteps, voxel_val)
                val_loss = nn.functional.mse_loss(noise_pred, noise)
                
            val_losses.append(val_loss.item())
    
    avg_val_loss = np.mean(val_losses)
    wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
    print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.6f}")

    # --- CHECKPOINTING ---
    # 1. Save Latest Model (Safety for resuming)
    torch.save(model.state_dict(), "exps/model_latest.pth")

    # 2. Save Best Model (If validation improved)
    if avg_val_loss < best_val_loss:
        print(f"--> Val Loss Improved ({best_val_loss:.6f} -> {avg_val_loss:.6f}). Saving Best Model.")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "exps/model_best.pth")

    # --- GENERATION SANITY CHECK ---
    # Visual check every N epochs
    # if (epoch + 1) % config["save_images_step"] == 0 or (epoch + 1) == config["epochs"]:
    #     print("Generating sanity check samples...")
    #     model.eval()
    #     num_samples_to_generate = min(5, len(val_dataset))

    #     for i in range(num_samples_to_generate):
    #         grasp_tensor, voxel, obj_id = val_dataset[i]
    #         voxel = voxel.unsqueeze(0).to(device) 

    #         # Start from pure noise
    #         sample = torch.randn(1, 19, device=device)

    #         with torch.no_grad():
    #             # Reverse diffusion loop
    #             for t in reversed(range(config["num_timesteps"])):
    #                 t_tensor = torch.tensor([t], device=device, dtype=torch.long)
    #                 residual = model(sample, t_tensor, voxel)
    #                 sample = noise_scheduler.step(residual, t_tensor, sample)

    #         # Denormalize
    #         grasp_np_norm = sample.cpu().numpy()[0]
    #         grasp_np = (grasp_np_norm + 1)/2 * (feature_max - feature_min) + feature_min

    #         generated_data.append((obj_id, grasp_np))

wandb.finish()

# ---- Final Save ----
torch.save(model.state_dict(), "exps/model_final.pth")
generated_data = np.array(generated_data, dtype=object)
np.save("exps/generated_grasps_with_id.npy", generated_data, allow_pickle=True)
np.save("exps/loss.npy", losses)

print("Training complete.")
print(f"Best Validation Loss: {best_val_loss:.6f}")
print("Models saved to exps/")



# #For inference
# # Load stats
# stats = np.load("exps/normalization_stats.npz")
# feature_min = stats["min"]
# feature_max = stats["max"]

# # ... Generate a grasp using the model ...
# prediction_normalized = model(...) 

# # Denormalize
# prediction_real = (prediction_normalized + 1)/2 * (feature_max - feature_min) + feature_min