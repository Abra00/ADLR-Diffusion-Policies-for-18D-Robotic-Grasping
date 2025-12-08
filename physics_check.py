# example call: python physics_check.py 02747177_1c3cf618a6790f1021c6005997c63924_0_generated_grasps.npz
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import argparse
import trimesh

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
    #calculate mass depending on volume with fixed density 
    mesh = trimesh.load(str(mesh_path))
    volume = mesh.convex_hull.volume  # get volume estimate
    print("Mesh-Volumen:", volume) # unit?
    rho = 100  # kg/m^3
    mass = volume * rho

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
        baseMass=mass,  
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
    # SIMULATION SETTINGS
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240) #timesteps  for each simulation 

    # iterate over grasps
    for i in range(len(positions)):
        pos = positions[i]
        rot = orientations[i]
        joint_vals = joints[i]
        print(f"\n=== Testing grasp {i+1}/{len(positions)} ===")

        # Set hand base position & orientation
        p.resetBasePositionAndOrientation(hand_id, pos, rot)

        # Set hand joint angles
        for k, j_idx in enumerate(ACTIVE_JOINTS):
            p.resetJointState(hand_id, j_idx, joint_vals[k])
            if j_idx in COUPLED_JOINTS:
                p.resetJointState(hand_id, j_idx+1, joint_vals[k])

        # small settling simulation (hand closes before gravity acts strongly)
        for _ in range(200):
            p.stepSimulation()

        # main test: does object stay in hand?
        stable = True
        start_pos, _ = p.getBasePositionAndOrientation(obj_id)  #only safe starting position not orientation 

        for step in range(int(3.0 / (1/240))): #run for 3 seconds 
            p.stepSimulation()

            cur_pos, _ = p.getBasePositionAndOrientation(obj_id)
            distance = np.linalg.norm(np.array(cur_pos) - np.array(start_pos))

            # if object dropped more than threshold → fail
            if distance > 0.005:  # 5 mm allowed: prevents false negatives caused by PyBullet's small simulation jitter.
                stable = False
                break

        if stable:
            print(f"Grasp {i} SUCCESS — object stayed in hand")
        else:
            print(f"Grasp {i} FAILED — object dropped")
        input()

    print("Finished evaluating all grasps.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Grasp npz filename in ./generated_grasps/")
    args = parser.parse_args()
    
    main(args.npz_file)