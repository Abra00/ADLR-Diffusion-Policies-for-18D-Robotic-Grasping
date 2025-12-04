import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from src.model import DiffusionMLP
from src.noise_scheduler import NoiseScheduler
from src.dataloader import load_and_process_data
from src.ema import EMA
import wandb
# Enable CuDNN benchmark for speed boost on fixed input sizes
torch.backends.cudnn.benchmark = True
# Fix for OpenMP error on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- CONFIGURATION ----
config = {
    "data_root" : "./Data/studentGrasping/processed_data_new",
    "batch_size": 64,
    "test_size": 0.1,
    "epochs": 200,
    "hidden_size": 512,
    "hidden_layers": 6,
    "save_dir": "./exps_new",
    "num_timesteps": 1000,
    "learning_rate": 2e-4,
    "resolution": 32,
    "embeding_size": 256, #time + voxel before 128
    "input_emb_dim": 64 #grasp
}



def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"Training on {device}...")

    data_root = config["data_root"]
    
    # 2. Prepare Data
    train_loader, val_loader, _, _ = load_and_process_data(
        data_dir=data_root,
        batch_size=config["batch_size"],
        test_size=config["test_size"]
    )

    noise_scheduler = NoiseScheduler(num_timesteps=config["num_timesteps"], device=device)

    model = DiffusionMLP(
        hidden_size=config["hidden_size"],
        num_layers=config["hidden_layers"],
        emb_size=config['embeding_size'],
        input_emb_dim=config['input_emb_dim']
    )
    model.to(device)

    # decay=0.999 means we average over roughly the last 1000 steps
    ema = EMA(model, decay=0.999)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config['epochs'],
        eta_min=1e-7
    )
    
    mse_loss = nn.MSELoss()

    scaler = torch.amp.GradScaler('cuda')

    # ---- THE TRAINING LOOP ----
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        batch_step = 0
        
        # Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch in pbar:

            grasps = batch[0].to(device)
            voxels = batch[1].to(device)
            
            # Generate random integers between 0 and 1000.
            # Shape should be (Batch_Size,).
            timesteps = torch.randint(0, config["num_timesteps"], size=(grasps.shape[0],), device=device).long()
            
            # Generate random Gaussian noise with same shape as 'grasps'
            noise = torch.randn_like(grasps).to(device)
            
            # Use the scheduler to combine grasps, noise, and t
            noisy_grasps = noise_scheduler.add_noise(grasps, noise, timesteps)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                noise_pred = model(noisy_grasps, timesteps, voxels)
                loss = mse_loss(noise_pred, noise)
                
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
                
            # Logging
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            # Proposed Logging Fix:
            global_step = epoch * len(train_loader) + batch_step
            if (batch_step + 1) % 50 == 0:
                wandb.log({"train/loss": loss.item(), "global_step": global_step})
            batch_step+=1
        
        avg_train_loss = train_loss/len(train_loader)
        wandb.log({"epoch_loss": avg_train_loss, "lr": train_scheduler.get_last_lr()[0], "epoch": epoch+1})
        train_scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                grasps_val = batch[0].to(device)
                voxels_val = batch[1].to(device)
                timesteps_val = torch.randint(0, config['num_timesteps'], (grasps_val.shape[0],), device=device).long()
                noise_val = torch.randn_like(grasps_val).to(device)
                noisy_grasps_val = noise_scheduler.add_noise(grasps_val, noise_val, timesteps_val)
                
                with torch.amp.autocast('cuda'):
                    noise_pred_val = model(noisy_grasps_val, timesteps_val, voxels_val)
                    val_loss = mse_loss(noise_pred_val, noise_val)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.6f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch+1})
        
        # Save Checkpoint every 10 epochs
        if (epoch + 1) % 20 == 0:
            # 1. Swap current weights for EMA weights
            ema.apply_shadow(model)
            
            # 2. Save the smooth model
            save_path = f"{config['save_dir']}/model_ep{epoch+1}_ema.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved smoothed checkpoint to {save_path}")
            
            # 3. Swap back to noisy weights to continue training math correctly
            ema.restore(model)

    torch.save(model.state_dict(), f"{config['save_dir']}/model_final.pth")
    print(f"Saved final model. Training is done")
    wandb.finish()

if __name__ == "__main__":
    wandb.init(
    project="adlr-diffusion_grasping",
    job_type="training_diff",
    config=config 
)
    train()