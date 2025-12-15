import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import gc

# --- CUSTOM IMPORTS ---
from src.model import DiffusionMLP
from src.noise_scheduler import NoiseScheduler
from src.dataloader_gce import load_and_process_data 
from src.ema import EMA
import wandb

# --- HARDWARE OPTIMIZATIONS ---
# Auf T4 wird TF32 ignoriert, schadet aber nicht.
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---- CONFIGURATION ----
config = {
    "wandb_key": "", # <--- BITTE EINTRAGEN!

    "data_root": "./Data/studentGrasping/processed_data", # <--- PFAD PRÜFEN!
    
    # --- PERFORMANCE TUNING ---
    "batch_size": 128,          
    "gradient_accumulation_steps": 1, 
    
    "test_size": 0.1,
    "epochs": 400,
    
    # --- MODEL ---
    "hidden_size": 512, 
    "hidden_layers": 6,
    "embeding_size": 256,
    "input_emb_dim": 64,
    
    "num_timesteps": 1000,
    "learning_rate": 2e-4, 
    "resolution": 32,
    "save_dir": "./exps_final_t4_size6_epoch400_batch256_scale2500"
}

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} (Streaming Mode)")
    
    # --- WANDB LOGIN ---
    if config["wandb_key"] and config["wandb_key"] != "DEIN_KEY_HIER":
        wandb.login(key=config["wandb_key"])
    
    wandb.init(project="adlr-diffusion_grasping", config=config)
    os.makedirs(config["save_dir"], exist_ok=True)

    # 2. Load Data (Loader Setup)
    print("Initializing DataLoaders...")
    # WICHTIG: Jetzt nutzen wir die Loader (train_loader, val_loader)
    train_loader, val_loader, _, _ = load_and_process_data(
        data_dir=config["data_root"],
        batch_size=config["batch_size"],
        test_size=config["test_size"]
    )

    # 3. Model Setup
    noise_scheduler = NoiseScheduler(num_timesteps=config["num_timesteps"], device=device)

    model = DiffusionMLP(
        hidden_size=config["hidden_size"],
        num_layers=config["hidden_layers"],
        emb_size=config['embeding_size'],
        input_emb_dim=config['input_emb_dim'],
        dropout=0.1
    )
    model.to(device)
    
    # Compile (Optional, T4 profitiert manchmal weniger, aber okay)
    print("Compiling Model...")
    try:
        model = torch.compile(model)
        print("Model compiled.")
    except Exception as e:
        print(f"Compile skipped: {e}")

    ema = EMA(model, decay=0.999)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    mse_loss = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() 

    # ---- TRAINING LOOP ----
    global_step = 0
    accum_steps = config["gradient_accumulation_steps"]
    
    print("--- Training Start ---")
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        # Standard Iterator über den Loader
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{config['epochs']}", mininterval=5.0)
        
        optimizer.zero_grad() 

        for step, batch in enumerate(pbar):
            # A. Daten auf GPU schieben (Streaming)
            # Dein Loader gibt zurück: grasp, voxel, id (id ignorieren wir hier)
            b_grasps = batch[0].to(device, non_blocking=True)
            b_voxels = batch[1].to(device, non_blocking=True)
            
            # B. Noise Generation
            timesteps = torch.randint(0, config["num_timesteps"], (b_grasps.shape[0],), device=device).long()
            noise = torch.randn_like(b_grasps)
            noisy_grasps = noise_scheduler.add_noise(b_grasps, noise, timesteps)
            
            # C. Forward Pass (Mixed Precision)
            with torch.cuda.amp.autocast():
                noise_pred = model(noisy_grasps, timesteps, b_voxels)
                loss = mse_loss(noise_pred, noise)
                loss = loss / accum_steps

            # D. Backward Pass
            scaler.scale(loss).backward()

            current_loss_val = loss.item() * accum_steps
            epoch_loss += current_loss_val

            # E. Optimizer Step
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                global_step += 1

                if global_step % 50 == 0:
                    wandb.log({"train/loss": current_loss_val, "global_step": global_step})
                    pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})

        # End of Epoch
        avg_train_loss = epoch_loss / len(train_loader)
        current_lr = train_scheduler.get_last_lr()[0]
        wandb.log({"epoch/train_loss": avg_train_loss, "epoch/lr": current_lr, "epoch": epoch+1})
        train_scheduler.step()

        # ---- VALIDATION LOOP ----
        model.eval()
        val_loss_accum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                b_grasps_val = batch[0].to(device, non_blocking=True)
                b_voxels_val = batch[1].to(device, non_blocking=True)
                
                t_val = torch.randint(0, config['num_timesteps'], (b_grasps_val.shape[0],), device=device).long()
                noise_val = torch.randn_like(b_grasps_val)
                noisy_val = noise_scheduler.add_noise(b_grasps_val, noise_val, t_val)
                
                with torch.cuda.amp.autocast():
                    pred_val = model(noisy_val, t_val, b_voxels_val)
                    v_loss = mse_loss(pred_val, noise_val)
                
                val_loss_accum += v_loss.item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        wandb.log({"epoch/val_loss": avg_val_loss, "epoch": epoch+1})

        # ---- SAVING ----
        if (epoch + 1) % 20 == 0:
            ema.apply_shadow(model)
            torch.save(model.state_dict(), f"{config['save_dir']}/model_ep{epoch+1}_ema.pth")
            ema.restore(model)

    # Final Save
    ema.apply_shadow(model)
    torch.save(model.state_dict(), f"{config['save_dir']}/model_final.pth")
    wandb.finish()
    print("Training finished successfully.")

if __name__ == "__main__":
    train()