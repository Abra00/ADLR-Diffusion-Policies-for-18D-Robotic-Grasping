# Usage example: python physics_check_test.py 

import pybullet 
import pybullet_data
import numpy as np
from pathlib import Path
import argparse
import trimesh


pybullet.connect(pybullet.GUI)

# Load hand
hand_id = pybullet.loadURDF(
    "C:/Gits/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/urdfs/dlr2.urdf",
    globalScaling=1,
    basePosition=[0, 0, 0],
    baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
)

# Load object
visualShapeId = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName="C:/Gits/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/mesh.obj",
                rgbaColor=[1,1,1,1],
                specularColor=[0.4, .4, 0],
                visualFramePosition=[0, 0, 0],
                meshScale=1)

collision_id = pybullet.createCollisionShape(
    shapeType=pybullet.GEOM_MESH,
    fileName="C:/Gits/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/mesh.obj",
    meshScale=1
)
object_id = pybullet.createMultiBody(
    baseMass=1,  
    baseInertialFramePosition=[0, 0, 0],
    baseCollisionShapeIndex=collision_id,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0,0,0],
    baseOrientation=[0,0,0,1]
)

                        
# Load grasps
data = np.load(Path("C:/Gits/ADLR-Diffusion-Policies-for-18D-Robotic-Grasping/Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/recording.npz"))

# Sort by grasp score
sorted_indx = np.argsort(data["scores"])[::-1]
print(data["grasps"].shape)
pybullet.setGravity(0, 0, -9.81)
pybullet.setTimeStep(1/240)
    # iterate over grasps
for i in sorted_indx:
    grasp = data["grasps"][i]

    # Set hand pose
    pybullet.resetBasePositionAndOrientation(bodyUniqueId=hand_id, posObj=grasp[:3], ornObj=grasp[3:7])
    
    # Set joint angles
    for k, j in enumerate([1,2,3, 7,8,9, 13,14,15, 19,20,21]):
        pybullet.resetJointState(hand_id, jointIndex=j, targetValue=grasp[7 + k], targetVelocity=0)
        # Set coupled joint
        if j in [3, 9, 15, 21]:
            pybullet.resetJointState(hand_id, jointIndex=j + 1, targetValue=grasp[7 + k], targetVelocity=0)

    # small settling simulation (hand closes before gravity acts strongly)
    for _ in range(200):
        pybullet.stepSimulation()

    # main test: does object stay in hand?
    stable = True
    start_pos, _ = pybullet.getBasePositionAndOrientation(object_id)  #only safe starting position not orientation 

    for step in range(int(3.0 / (1/240))): #run for 3 seconds 
        pybullet.stepSimulation()

        cur_pos, _ = pybullet.getBasePositionAndOrientation(object_id)
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
