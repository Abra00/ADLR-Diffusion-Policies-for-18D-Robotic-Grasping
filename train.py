import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb


try:
    from src.autoencoder_model import Autoencoder3D
except ImportError:
    print("Error: Could not import Autoencoder3D.")
    print("Make sure 'autoencoder_model.py' is inside a 'src' folder")
    print("and you are running this script from the project root.")
    exit()

# -------------------------------
# HYPERPARAMETERS
# -------------------------------
config = {
    "data_dir": "Data/studentGrasping/processed_meshes",
    "test_split": 0.25,
    "random_seed": 42,
    "latent_dim": 19,
    "epochs": 200,
    "batch_size": 32,  # !!! CRITICAL !!! Adjust based on GPU VRAM
    "learning_rate": 1.5e-3,
    "lr_scheduler_min": 1e-6,  # +++ ADDED +++ Minimum LR for scheduler
    "model_save_path": "models/autoencoder.pth",
    "encoder_save_path": "models/encoder_only.pth"
}

# -------------------------------
# 1. Initialize Weights & Biases
# -------------------------------
wandb.init(
    project="diffusion-grasping", # Same project as process_meshes
    job_type="training",          # This is a training run
    config=config                 # Save all hyperparameters
)
# Use the config values from wandb from now on
cfg = wandb.config

# -------------------------------
# 2. Custom PyTorch Dataset
# -------------------------------
class SDFDataset(Dataset):
    """Loads SDF .npy files from the processed data folder."""
    def __init__(self, root_dir):
        self.file_paths = []
        print(f"Loading data from: {root_dir}")
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".npy"):
                    self.file_paths.append(os.path.join(root, file))
        
        if not self.file_paths:
            print(f"Error: No .npy files found in {root_dir}.")
            print("Did you run 'process_meshes.py' first?")
            exit()
            
        print(f"Found {len(self.file_paths)} .npy files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            sdf = np.load(self.file_paths[idx])
        except Exception as e:
            print(f"Warning: Error loading {self.file_paths[idx]}: {e}. Skipping.")
            return None
            
        # Add a channel dimension
        tensor = torch.from_numpy(sdf).float().unsqueeze(0)
        return tensor

def collate_fn(batch):
    # This filters out None items (from loading errors)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# -------------------------------
# 3. Main Training Function
# -------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data (using cfg from wandb) ---
    full_dataset = SDFDataset(cfg.data_dir)
    
    train_paths, val_paths = train_test_split(
        full_dataset.file_paths, 
        test_size=cfg.test_split, 
        random_state=cfg.random_seed
    )
    
    # Create datasets from the split paths
    train_dataset = SDFDataset(cfg.data_dir)
    train_dataset.file_paths = train_paths
    
    val_dataset = SDFDataset(cfg.data_dir)
    val_dataset.file_paths = val_paths

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")
    wandb.log({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset)
    })

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    # --- Initialize model (using cfg from wandb) ---
    model = Autoencoder3D(latent_dim=cfg.latent_dim).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # +++ ADDED +++ Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,      # The number of epochs to complete one cosine cycle
        eta_min=cfg.lr_scheduler_min  # The minimum learning rate
    )

    # --- WANDB: Watch the model ---
    # This logs gradients and model topology
    wandb.watch(model, log="all", log_freq=100) # Log every 100 batches

    best_val_loss = float('inf')

    print("--- Starting Training ---")
    for epoch in range(cfg.epochs):
        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        for data in train_loader:
            if data is None: continue # Skip empty batches
            sdfs = data.to(device)
            reconstructions = model(sdfs)
            loss = loss_function(reconstructions, sdfs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                if data is None: continue # Skip empty batches
                sdfs = data.to(device)
                reconstructions = model(sdfs)
                loss = loss_function(reconstructions, sdfs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{cfg.epochs} \t Train Loss: {avg_train_loss:.6f} \t Val Loss: {avg_val_loss:.6f}")
        
        # --- WANDB: Log metrics ---
        # This sends the data to your online dashboard
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']  # +++ ADDED +++
        })

        scheduler.step()

        #Conditionally save the BEST model
        if val_loader and avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            # Save with a "_best" suffix
            best_model_path = cfg.model_save_path.replace(".pth", "_best.pth")
            best_encoder_path = cfg.encoder_save_path.replace(".pth", "_best.pth")
            
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.encoder.state_dict(), best_encoder_path)
            print(f"New best model saved! (Val Loss: {best_val_loss:.6f})")
            wandb.summary["best_val_loss"] = best_val_loss
        
        # Save the model in the last epoch(better for small testing)
        if epoch == (cfg.epochs-1):
            torch.save(model.state_dict(), cfg.model_save_path)
            torch.save(model.encoder.state_dict(), cfg.encoder_save_path)

    print("--- Training Complete ---")
    
    wandb.finish()


if __name__ == "__main__":
    train()