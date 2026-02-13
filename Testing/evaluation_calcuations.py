import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
#  Load Data
# =========================
with open("grasp_evaluation_results.json", "r") as f:
    data = json.load(f)

objects = data["objects"]
total_objects = len(objects)

# =========================
#  Compute statistics
# =========================

# 2a) Objects with no successful grasps
no_success_objects = sum(
    1 for obj in objects.values() if not any(grasp["success"] for grasp in obj["grasps"])
)
print(f"{(no_success_objects / total_objects) * 100:.2f}% of objects have no successful grasps")

# 2b) Objects with no grasps >= 3s
no_3s_objects = sum(
    1 for obj in objects.values() if not any(grasp.get("label") == ">=3s" for grasp in obj["grasps"])
)
print(f"{(no_3s_objects / total_objects) * 100:.2f}% of objects have no >=3s grasps")

# 2c) Collect volumes and grasp counts
volumes = []
total_grasps = []
success_grasps = []

for obj in objects.values():
    v = obj["volume"]
    grasps = obj["grasps"]
    total = len(grasps)
    success = sum(1 for g in grasps if g.get("label") == ">=3s")

    if total > 0:
        volumes.append(v)
        total_grasps.append(total)
        success_grasps.append(success)

volumes = np.array(volumes)
total_grasps = np.array(total_grasps)
success_grasps = np.array(success_grasps)

# =========================
#  Bin Data by Object Volume (log scale)
# =========================
bins = np.logspace(np.log10(volumes.min()), np.log10(volumes.max()), 10)
bin_indices = np.digitize(volumes, bins)

bin_centers = []
bin_success_rate = []
bin_grasp_count = []

for b in range(1, len(bins)):
    mask = bin_indices == b
    if not np.any(mask):
        continue

    total = total_grasps[mask].sum()
    success = success_grasps[mask].sum()

    bin_centers.append(np.sqrt(bins[b-1] * bins[b]))  # geometric mean
    bin_success_rate.append(success / total)
    bin_grasp_count.append(total)

# =========================
#  Configure Matplotlib for Poster
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

# Figure size (inches) for A0 poster-friendly layout
width_cm = 18  # adjust as needed
height_cm = 11
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))

# Scatter plot size proportional to number of grasps
sizes = np.array(bin_grasp_count) * 3

# Professional RGB color
color_rgb = (0.129, 0.318, 0.557)

# =========================
#  Plot Data
# =========================
scatter = ax.scatter(
    bin_centers,
    bin_success_rate,
    s=sizes,
    alpha=0.7,
    color=color_rgb
)

ax.plot(bin_centers, bin_success_rate, color=color_rgb, linewidth=2)  # line connecting points

# Logarithmic X-axis
ax.set_xscale("log")
ax.set_ylim(0, 1)

# Labels
ax.set_xlabel("Object Volume [m³]")
ax.set_ylabel("Grasp Success Rate (≥ 3s)")

# Grid for better readability
ax.grid(True, which="both", linestyle="--", alpha=0.4)

# Tight layout for saving
plt.tight_layout()

# =========================
#  Save for PowerPoint Poster
# =========================
plt.savefig("grasp_success_plot.png", dpi=300, bbox_inches="tight")
plt.show()
