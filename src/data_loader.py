import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---- URDF JOINT LIMITS ----
FINGER_MIN = [-0.523598, -0.349065, -0.174532]
FINGER_MAX = [ 0.523598,  1.500983,  1.832595]
# Replicate for 4 fingers
JOINTS_MIN = np.array(FINGER_MIN * 4, dtype=np.float32)
JOINTS_MAX = np.array(FINGER_MAX * 4, dtype=np.float32)

class GraspDataset(Dataset):
    def __init__(self, objects):
        self.items = []
        for obj in objects:
            voxel = obj["voxel_grid"]
            grasps = obj["grasps"]
            obj_id = obj["object_id"]
            center = obj["center"]
            scale = obj["scale"]
            
            for g in grasps:
                self.items.append({
                    "voxel": voxel,
                    "grasp": g,
                    "object_id": obj_id,
                    "center": center,
                    "scale": scale
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (torch.from_numpy(item["grasp"]).float(),
                torch.from_numpy(item["voxel"]).float(),
                item["object_id"])

def load_and_process_data(data_dir, batch_size, test_size=0.2):
    objects = []
    if not os.path.exists(data_dir): raise FileNotFoundError(f"Not found: {data_dir}")
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    for f in files:
        try:
            data = np.load(os.path.join(data_dir, f))
            voxel = data["voxel_sdf"].astype(np.float32)
            if len(voxel.shape) == 3: voxel = voxel.reshape(1, *voxel.shape)
            
            # RAW GRASPS (N, 19)
            raw_grasps = data["grasps"].astype(np.float32)
            
            # ---- NORMALIZE JOINTS ONLY (Indices 7-19) ----
            joints = raw_grasps[:, 7:]
            joints_norm = 2 * (joints - JOINTS_MIN) / (JOINTS_MAX - JOINTS_MIN + 1e-6) - 1
            
            final_grasps = raw_grasps.copy()
            final_grasps[:, 7:] = joints_norm

            objects.append({
                "voxel_grid": voxel,
                "grasps": final_grasps,
                "object_id": f.replace(".npz",""),
                "center": data["center"],
                "scale": data["scale"]
            })
        except: continue

    if not objects: raise ValueError("No data loaded!")

    # ---- FIX: CREATE 19D STATS TO MATCH MODEL OUTPUT ----
    # Pos (0-3) and Rot (3-7) are dummy 0/1 (we don't denormalize them with min/max)
    # Joints (7-19) are real limits
    full_min = np.zeros(19, dtype=np.float32)
    full_max = np.ones(19, dtype=np.float32)
    full_min[7:] = JOINTS_MIN
    full_max[7:] = JOINTS_MAX

# 1. First, split the data into train_objs and val_objs
    train_objs, val_objs = train_test_split(objects, test_size=test_size, random_state=42)
    
    # 2. NOW you can create the Datasets (because train_objs exists)
    train_dataset = GraspDataset(train_objs)
    val_dataset = GraspDataset(val_objs)

    # 3. Finally, wrap them in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=False, pin_memory=True)

    # 4. Return everything
    return train_loader, val_loader, train_dataset, val_dataset, full_min, full_max