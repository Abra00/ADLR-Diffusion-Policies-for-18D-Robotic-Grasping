# -*- coding: utf-8 -*-
# Usage example: python -m Testing.evaluation_testset
import pybullet 

import numpy as np
from pathlib import Path 
from PIL import Image
import pybullet_data
import trimesh
import json
import time
from tqdm import tqdm

def main():

    GRASPS_DIR = Path("Testing/generated_grasps")
    npz_files = list(GRASPS_DIR.glob("*.npz"))

    print(f"Visualizing {len(npz_files)} random files (first grasp only).")
    # CONNECT TO PYBULLET
    pybullet.connect(pybullet.DIRECT) 
    pybullet.setPhysicsEngineParameter(fixedTimeStep=1.0/1000.0, numSubSteps=2)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    #image setup
    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    IMG_ROOT = Path("Testing/grasp_images")
    IMG_ROOT.mkdir(exist_ok=True)
    #joint information
    COUPLED_JOINTS = [3,9,15,21]
    ACTIVE_JOINTS =  [1,2,3,7,8,9,13,14,15,19,20,21]
    # Good friction values for grasping
    FRICTION_HAND = 1.0         #friction value for hand with silicon 
    FRICTION_OBJECT = 0.5       #friction values for objects
    SPIN_F = 0.01               #reduce twisting slip
    ROLL_F = 0.0001
    #open values for joint 
    open_joint_values = [
        -0.5236,  # ringfinger_proximal 
        -0.3491,  # ringfinger_knuckle 
        -0.1745,  # ringfinger_middle 

        -0.5236,  # middlefinger_proximal 
        -0.3491,  # middlefinger_knuckle
        -0.1745,  # middlefinger_middle

        -0.5236,  # forefinger_proximal 
        -0.3491,  # forefinger_knuckle
        -0.1745,  # forefinger_middle

        -0.5236,  # thumb_proximal 
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
    results["meta"] = {
            "volume_unit": "m^3",
            "mass_unit": "kg",
            "density": 100,
            "volume_definition": "convex_hull"
        }
    #evaluation values
    dt = 1 / 1000
    max_time = 4.0
    MIN_VALID_HOLD = 0.065 #65ms (enough time for an object to drop 2 cm)
    SUCCESS_THRESHOLD = 3.0
    #controler values
    KP = 150.0   
    KD = 1.0    
    MAX_TORQUE = 1.0 
    for npz_path in tqdm(npz_files, desc="Evaluating objects"):
        print(f"Visualizing {npz_path.name} ...")
        #load grasp
        data = np.load(npz_path)
        positions = data["position"]      
        orientations = data["orientation"] 
        joints = data["joints"]          
        # get mesh path
        base_name = npz_path.stem.replace("_generated_grasps", "")
        obj_path = Path("Data/Testset/MultiGripperGrasp/GoogleScannedObjects") / base_name / "meshes/model_centered.obj"
        if not obj_path.exists():
            print(f"Mesh not found: {obj_path}")
        #calculate mass depending on volume with fixed density 
        mesh = trimesh.load(str(obj_path))
        volume = mesh.convex_hull.volume  # get volume estimate
        rho = 100  # kg/m^3
        mass = min(volume * rho,1) #set mass maximum 1 kg
        if base_name not in results["objects"]:
            results["objects"][base_name] = {
                "total_grasps": 0,
                "volume": float(volume),
                "mass": float(mass),
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

        print("Mesh-Volume:", volume) 
        print("Objekc-Mass:", mass, "kg")
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
                lateralFriction=FRICTION_HAND,
                spinningFriction=SPIN_F,
                rollingFriction=ROLL_F,
                contactStiffness=100000, 
                contactDamping=1000
            )

        # Set friction for object
        pybullet.changeDynamics(
            bodyUniqueId=object_id,
            linkIndex=-1,
            lateralFriction=FRICTION_OBJECT,
            spinningFriction=SPIN_F,
            rollingFriction=ROLL_F
        )  
        # --- AUTOMATIC CAMERA FOR HAND + OBJECT ---
        # Get all link IDs of the hand (including base link -1)
        hand_links = [-1] + list(range(pybullet.getNumJoints(hand_id)))

        # Collect all AABBs
        all_mins = []
        all_maxs = []

        # Object AABB
        obj_min, obj_max = pybullet.getAABB(object_id)
        all_mins.append(np.array(obj_min))
        all_maxs.append(np.array(obj_max))

        # Hand AABBs
        for link in hand_links:
            link_min, link_max = pybullet.getAABB(hand_id, link)
            all_mins.append(np.array(link_min))
            all_maxs.append(np.array(link_max))

        # Compute combined AABB
        aabb_min = np.min(np.array(all_mins), axis=0)
        aabb_max = np.max(np.array(all_maxs), axis=0)

        # Compute camera target and distance
        center = [(min_val + max_val) / 2 for min_val, max_val in zip(aabb_min, aabb_max)]
        size = np.linalg.norm(aabb_max - aabb_min)
        camera_distance = size * 1.5  # distance proportional to total size

        # Compute view and projection matrices
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=center,
            distance=camera_distance,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            fov=60,
            aspect=IMG_WIDTH / IMG_HEIGHT,
            nearVal=0.01,
            farVal=2.0
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
            for j_idx in ACTIVE_JOINTS:
                pybullet.setJointMotorControl2(
                    bodyIndex=hand_id,
                    jointIndex=j_idx,
                    controlMode=pybullet.VELOCITY_CONTROL,
                    force=0  # nullify internal motor
                )
            q_desired = joint_vals+0.1 #offest to get contact 
            pybullet.changeDynamics(object_id, -1, linearDamping=100, angularDamping=100) #make object hard to move
            # Close hand using torque-control PD loop
            for t in range(200):
                joint_states = pybullet.getJointStates(hand_id, ACTIVE_JOINTS)
                torques = []
                for idx, state in enumerate(joint_states):
                    q = state[0]  # joint position
                    qd = state[1] # joint velocity
                    tau = KP*(q_desired[idx]-q) - KD*qd
                    tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
                    torques.append(tau)
                pybullet.setJointMotorControlArray(
                    hand_id,
                    ACTIVE_JOINTS,
                    controlMode=pybullet.TORQUE_CONTROL,
                    forces=torques
                )
                pybullet.stepSimulation()
                #time.sleep(1/1000)
    
            print("Commanded torques:", torques)

            pybullet.changeDynamics(object_id, -1, linearDamping=0.04, angularDamping=0.04) # make object again normal to move 
            # Render Image before gravity 
            img = pybullet.getCameraImage(
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=pybullet.ER_TINY_RENDERER
            )
            rgb = np.reshape(img[2], (IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3].astype(np.uint8)
            #check for contact before simulation 
            contacts = pybullet.getContactPoints(bodyA=hand_id, bodyB=object_id)
            if len(contacts) == 0:
                hold_time = -1.0
                success = False
                label = "0s"
            else:
                # Enable gravity for realistic object behavior
                pybullet.setGravity(0,0,-9.81)

                start_pos, _ = pybullet.getBasePositionAndOrientation(object_id)  #only safe starting position not orientation 

                hold_time = 0.0

                for step in range(int(max_time / dt)):
                    joint_states = pybullet.getJointStates(hand_id, ACTIVE_JOINTS)
                    torques = []
                    for idx, state in enumerate(joint_states):
                        q = state[0]
                        qd = state[1]
                        tau = KP*(q_desired[idx]-q) - KD*qd
                        tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
                        torques.append(tau)
                    pybullet.setJointMotorControlArray(
                        hand_id,
                        ACTIVE_JOINTS,
                        controlMode=pybullet.TORQUE_CONTROL,
                        forces=torques
                    )
                    
                    pybullet.stepSimulation()
                    hold_time += dt #holdtime in seconds 
                    contacts = pybullet.getContactPoints(bodyA=hand_id, bodyB=object_id)
                    cur_pos, _ = pybullet.getBasePositionAndOrientation(object_id)
                    distance = np.linalg.norm(np.array(cur_pos) - np.array(start_pos))

                    # Stop if contact lost or moved too far
                    if len(contacts) == 0 or distance > 0.5:
                        break

            if hold_time < 0:
                label = "0s"
            elif hold_time >= 3.0:
                label = ">=3s"
            elif hold_time >= 2.0:
                label = ">=2s"
            elif hold_time >= 1.0:
                label = ">=1s"
            else:
                label = ">0s"
            success = hold_time >= SUCCESS_THRESHOLD
            # Save image 
            obj_img_dir = IMG_ROOT / base_name
            obj_img_dir.mkdir(exist_ok=True)

            status_str = "success" if success else "fail"
            img_path = obj_img_dir / f"grasp_{i:04d}_{status_str}.png"

            Image.fromarray(rgb).save(img_path)
            results["objects"][base_name]["grasps"].append({
                "grasp_id": int(i),
                "hold_time": float(round(hold_time, 4)),
                "label": label,
                "success": success,
                "image_path": str(img_path),
                "volume": float(volume),
                "mass": float(mass),
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
            print(f"hold time: {hold_time:.3f}s")



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
    print("\nSUCCESS RATE VS VOLUME (>=3s)")
    print("="*60)

    # volume vs grasp succes evalutaion 
    data = []
    for obj in results["objects"].values():
        for g in obj["grasps"]:
            success = 1 if g["hold_time"] >=SUCCESS_THRESHOLD else 0
            data.append((g["volume"], success))

    data = np.array(data)

    if len(data) == 0:
        print("No grasp data available for volume evaluation.")
    else:
        # log-bins
        bins = np.logspace(np.log10(data[:,0].min()),
                        np.log10(data[:,0].max()), 8)

        for i in range(len(bins)-1):
            mask = (data[:,0] >= bins[i]) & (data[:,0] < bins[i+1])
            if np.sum(mask) == 0:
                continue
            rate = np.mean(data[mask,1])
            print(f"{bins[i]:.2e} – {bins[i+1]:.2e} : "
                f"{rate*100:5.1f}%  ({np.sum(mask)} grasps)")


    print(f"Results saved to {OUTPUT_PATH}")
    print("Finished evaluating all grasps.")
if __name__ == "__main__":
    main()

