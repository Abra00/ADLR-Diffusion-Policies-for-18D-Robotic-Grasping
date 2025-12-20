import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class GraspDataset(Dataset):
    def __init__(self, objects_list):
        self.items = []
        
        print(f"Caching {len(objects_list)} objects to RAM... (This takes ~60s)")
        
        for obj in tqdm(objects_list, desc="Processing Tensors"):
            voxel_tensor = torch.from_numpy(obj['voxel_grid']).float()
            
            # Ensure correct shape (1, 32, 32, 32)
            if voxel_tensor.ndim == 3:
                voxel_tensor = voxel_tensor.unsqueeze(0)
            
            grasps = obj['grasps']
            obj_id = obj.get('object_id', 'unknown') # Handle potential missing ID
            
            # Convert each grasp to Tensor ONCE
            for g in grasps:
                g_tensor = torch.from_numpy(g).float()
                
                # Store the TENSORS in the list, not the raw dictionaries
                self.items.append({
                    'voxel': voxel_tensor, 
                    'grasp': g_tensor,
                    'obj_id': obj_id
                })
        
        print(f"Done. Dataset size: {len(self.items)} samples.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # --- CHANGE 2: Zero-Latency Access ---
        # Old code: Converted numpy->tensor here (Slow!)
        # New code: Just returns the pre-made tensor (Instant!)
        item = self.items[idx]
        
        # We return 3 items (includes ID for debugging)
        return item['grasp'], item['voxel'], item['obj_id']

def load_and_process_data(data_dir, batch_size, test_size=0.1):
    all_objects = []
    data_path = Path(data_dir)
    
    if not data_path.is_dir(): 
        raise FileNotFoundError(f"Not found: {data_dir}")

    files = sorted(list(data_path.glob("*.npz")))
    print(f"Found {len(files)} files. Loading from disk...")

    # Read from Disk
    for f in tqdm(files, desc="Disk I/O"):
        try: 
            data = np.load(str(f))
            all_objects.append({
                'voxel_grid': data['voxel_sdf'],
                'grasps': data['grasps'],
                'object_id': f.stem,
                'center': data['center'],
                'scale': data['scale']
            })
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue
            
    if not all_objects: raise ValueError("No data loaded!")

    train_objs, val_objs = train_test_split(all_objects, test_size=test_size, random_state=42)
    
    print("Preparing Training Set...")
    train_dataset = GraspDataset(train_objs)
    print("Preparing Validation Set...")
    val_dataset = GraspDataset(val_objs)
    
    # 5. Optimized DataLoaders
    # --- CHANGE 3: Performance Settings ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,           # Parallel Loading
        pin_memory=True,         # Faster CPU->GPU transfer
        persistent_workers=True, # Keeps workers alive between epochs
        prefetch_factor=2,       # Pre-loads batches so GPU never waits
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    return train_loader, val_loader, train_dataset, val_dataset