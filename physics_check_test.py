# -*- coding: utf-8 -*-
# Usage example: python physics_check_test.py 

import pybullet 

import numpy as np
from pathlib import Path

import time


pybullet.connect(pybullet.GUI)

# Load hand
hand_id = pybullet.loadURDF(
    "./Data/studentGrasping/urdfs/dlr2.urdf",
    globalScaling=1,
    basePosition=[0, 0, 0],
    baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
)

# Load object
visualShapeId = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName="./Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/mesh.obj",
                rgbaColor=[1,1,1,1],
                specularColor=[0.4, .4, 0],
                visualFramePosition=[0, 0, 0],
                meshScale=1)

collision_id = pybullet.createCollisionShape(
    shapeType=pybullet.GEOM_MESH,
    fileName="./Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/mesh.obj",
    meshScale=1,
    flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH
)
object_id = pybullet.createMultiBody(
    baseMass=1,  
    baseInertialFramePosition=[0, 0, 0],
    baseCollisionShapeIndex=collision_id,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0,0,0],
    baseOrientation=[0,0,0,1]
)

# Good friction values for grasping
FRICTION = 1.0       # radnom friction value
SPIN_F = 0.01         # reduce twisting slip
ROLL_F = 0.0001

# Set friction for all links of the hand
num_joints = pybullet.getNumJoints(hand_id)
for j in range(num_joints):
    pybullet.changeDynamics(
        bodyUniqueId=hand_id,
        linkIndex=j,
        lateralFriction=FRICTION,
        spinningFriction=SPIN_F,
        rollingFriction=ROLL_F
    )

# Set friction for object
pybullet.changeDynamics(
    bodyUniqueId=object_id,
    linkIndex=-1,
    lateralFriction=FRICTION,
    spinningFriction=SPIN_F,
    rollingFriction=ROLL_F
)                     
# Load grasps
data = np.load(Path("./Data/studentGrasping/student_grasps_v1/02747177/1c3cf618a6790f1021c6005997c63924/0/recording.npz"))

# Sort by grasp score
sorted_indx = np.argsort(data["scores"])[::-1]
print(data["grasps"].shape)
hand_joints = [1,2,3,7,8,9,13,14,15,19,20,21]
open_joint_values = [
    -0.5236,  # ringfinger_proximal -> nach aussen
    -0.3491,  # ringfinger_knuckle -> leicht gestreckt
    -0.1745,  # ringfinger_middle -> leicht gestreckt

    -0.5236,  # middlefinger_proximal -> nach aussen
    -0.3491,  # middlefinger_knuckle
    -0.1745,  # middlefinger_middle

    -0.5236,  # forefinger_proximal -> nach aussen
    -0.3491,  # forefinger_knuckle
    -0.1745,  # forefinger_middle

    -0.5236,  # thumb_proximal -> nach aussen
    -0.3491,  # thumb_knuckle
    -0.1745,  # thumb_middle
]
    # iterate over grasps
hand_joints = [1,2,3,7,8,9,13,14,15,19,20,21]


#final joints of each finger 
final_joints_indices_in_hand_joints = [2, 5, 8, 11]

for i in sorted_indx:
    grasp = data["grasps"][i]

    #set object back:
    pybullet.resetBasePositionAndOrientation(
    bodyUniqueId=object_id,
    posObj=[0,0,0],        
    ornObj=[0,0,0,1]       
    )
    pybullet.resetBaseVelocity(object_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])

    # Set hand pose
    joint_values = grasp[7:19]
    # Temporarily disable gravity
    pybullet.setGravity(0,0,0)
    # set starting positon 
    pybullet.resetBasePositionAndOrientation(hand_id, posObj=grasp[:3], ornObj=grasp[3:7])
    for k, j_idx in enumerate(hand_joints):
        pybullet.resetJointState(hand_id, jointIndex=j_idx, targetValue=open_joint_values[k], targetVelocity=0)
        # copplet joints
        if j_idx in [3, 9, 15, 21]:
            pybullet.resetJointState(hand_id, jointIndex=j_idx+1, targetValue=open_joint_values[k], targetVelocity=0)
    print("start set")
    # ------------------------
    # Move hand to grasp pose (Position-Control)
    # ------------------------
    q_desired = joint_values+0.2 #offest to get contact 
    pybullet.setJointMotorControlArray(
        hand_id, hand_joints,
        controlMode=pybullet.POSITION_CONTROL,
        targetPositions=q_desired,
        forces=[1]*len(hand_joints)  # N·m
    )
    for x in range(500):  # let the hand grap first 
        pybullet.stepSimulation()
        print("grasping")
        y=pybullet.getJointStates(hand_id,hand_joints)
        print([j[-1] for j in y]) 
        


    print("grasping")
    y=pybullet.getJointStates(hand_id,hand_joints)
    print([j[-1] for j in y]) 
    pybullet.setGravity(0,0,-9.81)
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
