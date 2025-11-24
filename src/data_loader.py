import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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

def load_and_process_data(voxel_dir, grasp_dir, batch_size, test_size=0.2):
    """
    Loads .npy and .npz files, normalizes grasps, creates datasets and dataloaders.
    Returns: train_loader, val_loader, train_dataset, val_dataset, feature_min, feature_max
    """
    objects = []
    
    # List all voxel files
    if not os.path.exists(voxel_dir):
        raise FileNotFoundError(f"Voxel directory not found: {voxel_dir}")
        
    voxel_files = sorted([f for f in os.listdir(voxel_dir) if f.endswith(".npy")])
    print(f"Found {len(voxel_files)} voxel files.")

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

    if len(objects) == 0:
        raise ValueError("No valid objects loaded. Check your data paths.")

    # Concatenate all grasps to compute min/max for normalization
    all_grasps = np.concatenate([obj["grasps"] for obj in objects], axis=0)
    feature_min = all_grasps.min(axis=0)  # Per-feature minimum
    feature_max = all_grasps.max(axis=0)  # Per-feature maximum

    # ---- Normalize grasps to [-1, 1] ----
    for obj in objects:
        obj["grasps"] = 2 * (obj["grasps"] - feature_min) / (feature_max - feature_min + 1e-6) - 1

    # ---- Split data into train and validation sets ----
    train_objs, val_objs = train_test_split(objects, test_size=test_size, random_state=42)
    train_dataset = GraspDataset(train_objs)
    val_dataset = GraspDataset(val_objs)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=8,
                              persistent_workers=True, 
                              drop_last=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=4,
                            persistent_workers=True, 
                            drop_last=False)

    return train_loader, val_loader, train_dataset, val_dataset, feature_min, feature_max