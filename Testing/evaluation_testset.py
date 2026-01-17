# -*- coding: utf-8 -*-
# Usage example: python -m Testing.evaluation_testset
import pybullet 

import numpy as np
from pathlib import Path
import pybullet_data
import trimesh
import json
import time

def main():

    GRASPS_DIR = Path("Testing/generated_grasps")
    npz_files = list(GRASPS_DIR.glob("*.npz"))

    print(f"Visualizing {len(npz_files)} random files (first grasp only).")
    # CONNECT TO PYBULLET
    pybullet.connect(pybullet.GUI) 
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    #joint information
    COUPLED_JOINTS = [3,9,15,21]
    ACTIVE_JOINTS =     [1,2,3,7,8,9,13,14,15,19,20,21]
    # Good friction values for grasping
    FRICTION = 1.0       # radnom friction value
    SPIN_F = 0.01         # reduce twisting slip
    ROLL_F = 0.0001
    #open values for joint 
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
    #variable to store evaluation
    results = {
        "summary": {
            "total_objects": 0,
            "total_grasps": 0,
            "successful_grasps": {
                ">=3s": 0,
                ">=2s": 0,
                ">=1s": 0,
                ">0s": 0,
                "0s": 0
            }
        },
        "objects": {}
    }
    #evaluation values
    dt = 1 / 240
    max_time = 3.0
    hold_time = 0.0
    MIN_VALID_HOLD = 0.05 #50ms
    for npz_path in npz_files:
        print(f"Visualizing {npz_path.name} ...")
        #load grasp
        data = np.load(npz_path)
        positions = data["position"]      
        orientations = data["orientation"] 
        joints = data["joints"]          
        # get mesh path
        base_name = npz_path.stem.replace("_generated_grasps", "")
        obj_path = Path("Data/Testset/MultiGrippperGrasp/GoogleScannedObjects") / base_name / "meshes/model_centered.obj"
        if not obj_path.exists():
            print(f"Mesh not found: {obj_path}")
        if base_name not in results["objects"]:
            results["objects"][base_name] = {
                "total_grasps": 0,
                "successful_grasps": {
                    ">=3s": 0,
                    ">=2s": 0,
                    ">=1s": 0,
                    ">0s": 0,
                    "0s": 0
                },
                "grasps": []
            }
            results["summary"]["total_objects"] += 1
        #calculate mass depending on volume with fixed density 
        mesh = trimesh.load(str(obj_path))
        volume = mesh.convex_hull.volume  # get volume estimate
        print("Mesh-Volumen:", volume) # unit?
        rho = 100  # kg/m^3
        mass = min(volume * rho,1) #set mass maximum 1 kg
        print("Objekt-Masss:", mass, "kg")
        # delete old objects
        pybullet.resetSimulation()
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

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
                        fileName=str(obj_path),
                        rgbaColor=[1,1,1,1],
                        specularColor=[0.4, .4, 0],
                        visualFramePosition=[0, 0, 0],
                        meshScale=1)

        collision_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=str(obj_path),
            meshScale=1,
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH
        )
        object_id = pybullet.createMultiBody(
            baseMass=mass,  
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0,0,0],
            baseOrientation=[0,0,0,1]
        )
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
        for i in range(len(positions)):
            pos = positions[i]
            rot = orientations[i]
            joint_vals = joints[i]

            # Temporarily disable gravity
            pybullet.setGravity(0,0,0)

            # Set hand base position and orientation for current grasp
            pybullet.resetBasePositionAndOrientation(hand_id, pos, rot)
            # Reset object to start position and orientation
            #set object back:
            pybullet.resetBasePositionAndOrientation(
            bodyUniqueId=object_id,
            posObj=[0,0,0],        
            ornObj=[0,0,0,1]       
            )
            #reset velocity 
            pybullet.resetBaseVelocity(object_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
            # --- Set hand to OPEN position instantly (no physics) ---
            for k, j_idx in enumerate(ACTIVE_JOINTS):
                pybullet.resetJointState(
                    bodyUniqueId=hand_id,
                    jointIndex=j_idx,
                    targetValue=open_joint_values[k],
                    targetVelocity=0
                )
                # Coupled joints: apply same open value
                if j_idx in COUPLED_JOINTS:
                    pybullet.resetJointState(
                        bodyUniqueId=hand_id,
                        jointIndex=j_idx + 1,
                        targetValue=open_joint_values[k],
                        targetVelocity=0
                    )
            # ------------------------
            # Move hand to grasp pose (Position-Control)
            # ------------------------
            q_desired = joint_vals+0.1 #offest to get contact 
            pybullet.setJointMotorControlArray(
                hand_id, ACTIVE_JOINTS,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=q_desired,
                forces=[1]*len(ACTIVE_JOINTS)  # N·m
            )

            for x in range(200):  # let the hand grap first 
                pybullet.stepSimulation()
                #time.sleep(1/240) # only visualisation
    
            print("Final grasp strenght")
            y=pybullet.getJointStates(hand_id,ACTIVE_JOINTS)
            print([j[-1] for j in y])   
            # Enable gravity for realistic object behavior
            pybullet.setGravity(0,0,-9.81)

            start_pos, _ = pybullet.getBasePositionAndOrientation(object_id)  #only safe starting position not orientation 

            dt = 1 / 240
            max_time = 3.0
            hold_time = 0.0

            start_pos, _ = pybullet.getBasePositionAndOrientation(object_id)

            for step in range(int(max_time / dt)):
                pybullet.stepSimulation()
                cur_pos, _ = pybullet.getBasePositionAndOrientation(object_id)
                distance = np.linalg.norm(np.array(cur_pos) - np.array(start_pos))

                if distance > 0.005: #if objects movs more than 5 mm 
                    break

                hold_time += dt #holdtime in seconds 
            if hold_time < MIN_VALID_HOLD:
                label = "0s"
            elif hold_time >= 3.0:
                label = ">=3s"
            elif hold_time >= 2.0:
                label = ">=2s"
            elif hold_time >= 1.0:
                label = ">=1s"
            else:
                label = ">0s"
            results["objects"][base_name]["grasps"].append({
                "grasp_id": int(i),
                "hold_time": float(round(hold_time, 4)),
                "label": label,
                "joint_values": joint_vals.tolist(),
                "position": pos.tolist(),
                "orientation": rot.tolist()
            })
            # object statistic
            obj_stats = results["objects"][base_name]
            obj_stats["total_grasps"] += 1
            obj_stats["successful_grasps"][label] += 1

            # global statistic
            results["summary"]["total_grasps"] += 1
            results["summary"]["successful_grasps"][label] += 1

    OUTPUT_PATH = Path("Testing/grasp_evaluation_results.json")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    #console output
    print("\n" + "="*60)
    print("GRASP EVALUATION SUMMARY")
    print("="*60)

    summary = results["summary"]

    print(f"Total objects evaluated : {summary['total_objects']}")
    print(f"Total grasps evaluated  : {summary['total_grasps']}\n")

    print("Grasp success distribution:")
    for label, count in summary["successful_grasps"].items():
        percentage = (
            100.0 * count / summary["total_grasps"]
            if summary["total_grasps"] > 0 else 0.0
        )
        print(f"  {label:>4} : {count:6d} grasps ({percentage:6.2f} %)")

    success_3s = summary["successful_grasps"][">=3s"]
    success_rate = (
        100.0 * success_3s / summary["total_grasps"]
        if summary["total_grasps"] > 0 else 0.0
    )

    print("\nStrict success rate (>=3s hold): "
        f"{success_rate:.2f} %")
    print("="*60 + "\n")

    print(f"Results saved to {OUTPUT_PATH}")
    print("Finished evaluating all grasps.")
if __name__ == "__main__":
    main()

