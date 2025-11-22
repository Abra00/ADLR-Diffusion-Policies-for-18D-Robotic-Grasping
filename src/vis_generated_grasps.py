# use the following command for example : python "/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/src/vis_generated_grasps.py" \
"/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/exps/generated_grasps_with_id.npy" \
"/Users/lucafrontini/Library/Mobile Documents/com~apple~CloudDocs/Uni/TUM/2. Semester /Advanced Deep Learning for Robotics/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1"
from pathlib import Path
import pybullet
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Visualize 19D grasps for objects")
parser.add_argument("data_file", help="Pfad zur npy/npz Datei mit (Objekt-ID, 19D-Griff) Pairs")
parser.add_argument("mesh_root", help="Root-Ordner, in dem die Objekt-Meshes liegen")
args = parser.parse_args()

pybullet.connect(pybullet.GUI)

# Lade Hand-URDF
hand_id = pybullet.loadURDF(
    "Data/studentGrasping/urdfs/dlr2.urdf",
    globalScaling=1,
    basePosition=[0, 0, 0],
    baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
)

# Lade deine Daten
data = np.load(args.data_file, allow_pickle=True)

# Prüfen, ob es NPZ-Container oder Array ist
if isinstance(data, np.lib.npyio.NpzFile):
    pairs = []
    for key in data.files:
        pairs.extend(data[key].tolist())
else:
    pairs = data.tolist()

# Listen für Objekt-IDs und Griffe
object_ids = []
grasps = []
for obj_id, grasp in pairs:
    object_ids.append(obj_id)
    grasps.append(grasp)

grasps = np.stack(grasps)  # shape: (num_samples, 19)

# Mesh root
mesh_root = Path(args.mesh_root)

# Visualisiere jeden Griff
for obj_id, grasp in zip(object_ids, grasps):
    # Pfad zum Mesh zusammensetzen
    folders = obj_id.split("_")
    mesh_path = mesh_root.joinpath(*folders) / "mesh.obj"
    if not mesh_path.exists():
        print(f"Mesh nicht gefunden für Objekt {obj_id}: {mesh_path}")
        continue

    visualShapeId = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=str(mesh_path),
        rgbaColor=[1,1,1,1],
        specularColor=[0.4, 0.4, 0],
        visualFramePosition=[0, 0, 0],
        meshScale=1
    )
    object_id = pybullet.createMultiBody(
        baseMass=1,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=visualShapeId,
        baseCollisionShapeIndex=visualShapeId,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1]
    )

    # Hand Position & Orientation
    pybullet.resetBasePositionAndOrientation(
        bodyUniqueId=hand_id,
        posObj=grasp[:3],
        ornObj=grasp[3:7]
    )

    # Setze Gelenkwinkel (19D Griff)
    joint_indices = [1,2,3, 7,8,9, 13,14,15, 19,20,21]
    for k, j in enumerate(joint_indices):
        pybullet.resetJointState(hand_id, jointIndex=j, targetValue=grasp[7 + k], targetVelocity=0)
        # Setze gekoppelte Gelenke
        if j in [3, 9, 15, 21]:
            pybullet.resetJointState(hand_id, jointIndex=j + 1, targetValue=grasp[7 + k], targetVelocity=0)

    print(f"Objekt {obj_id}, Griff visualisiert.")
    input("Drücke Enter für nächsten Griff...")

    # Objekt entfernen, um Platz für den nächsten zu machen
    pybullet.removeBody(object_id)
