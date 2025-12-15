import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GraspDataset(Dataset):
    def __init__(self, objects_list):
        self.items = []
        for obj in objects_list:
            # Wir speichern Referenzen. 
            # Da 'load_and_process_data' schon alles in den RAM geladen hat,
            # greifen wir hier nur darauf zu.
            voxel = obj['voxel_grid']
            grasps = obj['grasps']
            center = obj['center']
            scale = obj['scale']
            
            for g in grasps:
                self.items.append({
                    'voxel' : voxel,
                    'grasp' : g,
                    'center': center,
                    'scale' : scale,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Optimierung: .float() ist teuer, wenn es oft passiert. 
        # Da wir die Daten schon als float32 laden, reicht torch.from_numpy oft aus.
        return (torch.from_numpy(item['grasp']),
                torch.from_numpy(item['voxel']))

# --- KORREKTUR HIER ---
# Wir fügen num_workers als Parameter hinzu!
def load_and_process_data(data_dir, batch_size, test_size=0.1, num_workers=6):
    all_objects = []
    data_path = Path(data_dir)
    if not data_path.is_dir() : raise FileNotFoundError(f"Not found: {data_dir}")

    files = sorted(list(data_path.glob("*.npz")))

    # HINWEIS: Dieser Teil läuft IMMER auf nur einem Kern (Main Process)
    # Das erklärt, warum am Anfang nur 1 Kern arbeitet.
    print(f"Lade {len(files)} Dateien in den RAM...")
    
    for f in files:
        try:
            data = np.load(f)
            if len(data['voxel_sdf'].shape) == 3: 
                voxel = data['voxel_sdf'].reshape(1, *data['voxel_sdf'].shape)
            
            # WICHTIG: Hier schon sicherstellen, dass es float32 ist
            all_objects.append({
                'voxel_grid': voxel.astype(np.float32),
                'grasps' : data['grasps'].astype(np.float32), # Auch hier sicherstellen
                'center' : data['center'],
                'scale' : data['scale']
            })
        except Exception as e:
            print(f'Problem during loading of object {f}: {e}')
            continue
            
    if not all_objects:
        raise ValueError("No data loaded!")

    # Create dataset
    train_objs, val_objs = train_test_split(
        all_objects,
        test_size=test_size,
        random_state=69
    )
    
    train_dataset = GraspDataset(train_objs)
    val_dataset = GraspDataset(val_objs)
    
    print(f"Erstelle DataLoader mit num_workers={num_workers}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Hier nutzen wir jetzt die Variable
        num_workers=num_workers,
        persistent_workers=True, 
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Val braucht meist kein Shuffle
        num_workers=2, # Validation braucht weniger Worker
        persistent_workers=True,
        drop_last=False,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset