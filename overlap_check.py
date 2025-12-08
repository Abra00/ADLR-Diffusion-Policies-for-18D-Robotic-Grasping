# example call: python overlap_check.py 02747177_1c3cf618a6790f1021c6005997c63924_0_generated_grasps.npz
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import argparse

COUPLED_JOINTS = [3,9,15,21]
ACTIVE_JOINTS = [1,2,3,7,8,9,13,14,15,19,20,21]
def main(grasps):
    urdf_path="./Data/studentGrasping/urdfs/dlr2.urdf"
    #get mesh location
    base_dir = Path("./Data/studentGrasping/student_grasps_v1")
    grasps = Path(grasps)  # convert string filename to Path
    basename = grasps.stem 
    object_str = basename.replace("_generated_grasps", "") # Remove '_generated_grasps'
    parts = object_str.split("_", 1) # Split on the first underscore to get folder structure
    folder_path = Path(parts[0]) / Path(parts[1].replace("_", "/"))
    mesh_path = base_dir / folder_path / "mesh.obj"
    # CONNECT TO PYBULLET
    p.connect(p.GUI)  #change to p.direct when no visualization is needed 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    #load object
    visualShapeId = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(mesh_path),
        rgbaColor=[1,1,1,1],
        specularColor=[0.4,0.4,0],
        visualFramePosition=[0,0,0],
        meshScale=1
    )
    collision_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(mesh_path),
        meshScale=[1,1,1]
    )
    obj_id = p.createMultiBody(
        baseMass=1,  
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visualShapeId,
        basePosition=[0,0,0],
        baseOrientation=[0,0,0,1]
    )



    # LOAD HAND
    hand_id = p.loadURDF(
        urdf_path,
        globalScaling=1,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_MAINTAIN_LINK_ORDER,
    )

    # LOAD GRASPS
    grasp_dir = Path("./generated_grasps") # Folder where grasps are stored
    grasp_path = grasp_dir/ grasps.name
    data = np.load(grasp_path)
    positions = data["position"]      # (N,3)
    orientations = data["orientation"] # (N,4)
    joints = data["joints"]           # (N,12)

    # CHECK COLLISION / OVERLAP
    results = []
    penetration_tolerance = 0.005 #5 mm

    for i in range(len(positions)):
        pos = positions[i]
        rot = orientations[i]
        joint_vals = joints[i]

        # Set hand base position & orientation
        p.resetBasePositionAndOrientation(hand_id, pos, rot)

        # Set hand joint angles
        for k, j_idx in enumerate(ACTIVE_JOINTS):
            p.resetJointState(hand_id, j_idx, joint_vals[k])
            if j_idx in COUPLED_JOINTS:
                p.resetJointState(hand_id, j_idx+1, joint_vals[k])

        # Check for penetration / collision
        contacts = p.getClosestPoints(hand_id, obj_id, distance=penetration_tolerance)
        overlaps = any([c[8] < -penetration_tolerance for c in contacts])

        # Check number of contact points
        total_contacts = 0
        for link in ACTIVE_JOINTS:
            link_contacts = p.getClosestPoints(hand_id, obj_id, distance=0.0, linkIndexA=link)
            total_contacts += len(link_contacts)
        enough_contacts = total_contacts >= 2 #at least 2 contact points 

        # Determine if grasp is valid
        if overlaps and not enough_contacts:
            print(f"Grasp {i} invalid: Penetration AND less than 2 contact points ({total_contacts})")
        elif overlaps:
            print(f"Grasp {i} invalid: Penetration only")
        elif not enough_contacts:
            print(f"Grasp {i} invalid: less than 2 contact points ({total_contacts})")
        else:
            print(f"Grasp {i} is valid (collision-free, ≥2 contacts)")
        input() #dont use if you dont want to visualize 

    # SUMMARY
    num_overlaps = sum(results)
    num_free = len(results) - num_overlaps
    print(f"\nTotal grasps: {len(results)}")
    print(f"Overlapping grasps: {num_overlaps}")
    print(f"Collision-free grasps: {num_free}")

    # Disconnect from PyBullet
    p.disconnect()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Grasp npz filename in ./generated_grasps/")
    args = parser.parse_args()
    
    main(args.npz_file)