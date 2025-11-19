import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from src.Diff_model import MLPWithVoxel, NoiseScheduler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameter ----
config = {
    "batch_size": 128,
    "eval_batch_size": 1000,
    "epochs": 20,
    "learning_rate": 1e-3,
    "num_timesteps": 100,
    "hidden_size": 128,
    "hidden_layers": 3,
    "embedding_size": 128,
    "save_images_step": 50
}
# ---- Load Data ----
voxel_dir = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_meshes"
grasp_dir = grasp_dir = "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/processed_scores"

objects = []

# Liste aller Voxel-Dateien
voxel_files = sorted([f for f in os.listdir(voxel_dir) if f.endswith(".npy")])

for vf in voxel_files:
    voxel_path = os.path.join(voxel_dir, vf)
    grasp_path = os.path.join(grasp_dir, vf.replace(".npy", ".npz"))
    
    if not os.path.exists(grasp_path):
        print(f"Warnung: Kein Grasp zu {vf} gefunden, überspringe.")
        continue

    voxel = np.load(voxel_path).astype(np.float32)        # (32,32,32)
    voxel = voxel.reshape(1, 32, 32, 32)                  # Kanal hinzufügen

    grasp_data = np.load(grasp_path)
    grasps = grasp_data["grasps"].astype(np.float32)      # (N,19)

    objects.append({
        "voxel_grid": voxel,
        "grasps": grasps,
        "object_id": vf.replace(".npy","")
    })

print("Geladene Objekte:", len(objects))
# Dataset Class
# -------------------------------
class GraspDataset(Dataset):
    def __init__(self, objects):
        self.items = []
        for obj in objects:
            voxel = obj["voxel_grid"]        # (1,32,32,32)
            grasps = obj["grasps"]           # (K,19)
            obj_id = obj["object_id"]
            for g in grasps:
                self.items.append({
                    "voxel": voxel,
                    "grasp": g,
                    "object_id": obj_id
                })


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (torch.from_numpy(item["grasp"]).float(),
                torch.from_numpy(item["voxel"]).float(),
                item["object_id"])




train_objs, val_objs = train_test_split(objects, test_size=0.2)

train_dataset = GraspDataset(train_objs)
all_voxels = torch.tensor(np.concatenate([obj["voxel_grid"][None] for obj in objects], axis=0))
val_dataset = GraspDataset(val_objs)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# ---- Modell & Optimizer ----
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

# ---- Training Loop ----
losses = []
generated_data = []


for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{config["epochs"]}]')
    for i, (x_batch, voxel_batch, obj_id_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)          # (B,19)
        voxel_batch = voxel_batch.to(device)  # (B,1,32,32,32)
        
        noise = torch.randn_like(x_batch)     # Noise für alle 19D-Griffe
        timesteps = torch.randint(0, config["num_timesteps"], (x_batch.shape[0],), device=device)
        
        noisy = noise_scheduler.add_noise(x_batch, noise, timesteps)
        
        # Forward
        noise_pred = model(noisy, timesteps, voxel_batch)
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name} hat keinen Gradienten!")
            else:
                print(f"{name} grad mean:", param.grad.abs().mean().item())
        optimizer.step()
        optimizer.zero_grad() 
        losses.append(loss.item())
        epoch_loss += loss.item()
        train_loader_tqdm.set_postfix(curr_train_loss=f"{epoch_loss / (i + 1):.8f}")
    scheduler.step()
    print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss / len(train_loader):.6f}")
    # ---- Validation ----
    model.eval()
    val_losses = []
    validation_loss = 0
    validation_loader_tqdm = tqdm(val_loader, desc=f'Validation Epoch [{epoch + 1}/{config["epochs"]}]')
    with torch.no_grad():
        for i, (x_val, voxel_val) in enumerate(val_loader):
            x_val = x_val.to(device)
            voxel_val = voxel_val.to(device)

            noise = torch.randn_like(x_val)
            timesteps = torch.randint(0, config["num_timesteps"], (x_val.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(x_val, noise, timesteps)
            
            noise_pred = model(noisy, timesteps, voxel_val)
            val_loss = nn.functional.mse_loss(noise_pred, noise)
            val_losses.append(val_loss.item())
            validation_loss += val_loss.item()
            validation_loader_tqdm.set_postfix(val_loss = "{:.8f}".format(validation_loss / (i + 1)))
    avg_val_loss = np.mean(val_losses)
    # Samples generieren
    if epoch % config["save_images_step"] == 0 or epoch == config["epochs"]-1:
        model.eval()

        num_samples_to_generate = min(5, len(val_dataset))
        for i in range(num_samples_to_generate):
            grasp_tensor, voxel, obj_id = val_dataset[i]
            voxel = voxel.unsqueeze(0).to(device)  # Batch dimension
            sample = torch.randn(1, 19, device=device)  # Start mit Noise

            # Rückwärtsdiffusion
            for t in reversed(range(config["num_timesteps"])):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                with torch.no_grad():
                    residual = model(sample, t_tensor, voxel)
                sample = noise_scheduler.step(residual, t, sample)

            # Objekt-ID + Griff zusammen speichern
            grasp_np = sample.cpu().numpy()[0]  # 19D
            generated_data.append((obj_id, grasp_np))


# ---- Ergebnisse speichern ----
generated_data = np.array(generated_data, dtype=object)
os.makedirs("exps", exist_ok=True)
torch.save(model.state_dict(), "exps/model.pth")
np.save("exps/generated_grasps_with_id.npy", np.array(generated_data, dtype=object), allow_pickle=True)
np.save("exps/loss.npy", losses)

