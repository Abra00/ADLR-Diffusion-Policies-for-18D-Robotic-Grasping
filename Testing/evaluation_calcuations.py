import json

with open("grasp_evaluation_results.json", "r") as f:
    data = json.load(f)

objects = data["objects"]
total_objects = len(objects)

# 1) Kein einziger success=True
no_success_objects = 0
for obj in objects.values():
    grasps = obj["grasps"]
    if not any(grasp["success"] is True for grasp in grasps):
        no_success_objects += 1

print(f"{(no_success_objects / total_objects) * 100:.2f}% der Objekte haben keinen einzigen erfolgreichen Griff")

# 2) Kein einziger >=3s
no_3s_objects = 0
for obj in objects.values():
    grasps = obj["grasps"]
    if not any(grasp.get("label") == ">=3s" for grasp in grasps):
        no_3s_objects += 1

print(f"{(no_3s_objects / total_objects) * 100:.2f}% der Objekte haben keinen >=3s-Griff")
