import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# FILE PATHS
# ==========================================================
pred_path = "/DATA/Ego4d/episodic-memory/NLQ/2D-TAN/nlq_2dtan_submission_og_2dtan.json"
gt_path   = "/DATA/Ego4d/Ego4d_data/v2/annotations/nlq_all.json"

save_dir = Path("./analysis_plots")
save_dir.mkdir(exist_ok=True)

# ==========================================================
# LOAD FILES
# ==========================================================
with open(pred_path, "r") as f:
    pred_data = json.load(f)

with open(gt_path, "r") as f:
    gt_data = json.load(f)

# ==========================================================
# BUILD GT LOOKUP
# ==========================================================
gt_lookup = {}

for video in gt_data["videos"]:
    for clip in video["clips"]:
        clip_uid = clip["clip_uid"]

        for ann in clip["annotations"]:
            ann_uid = ann["annotation_uid"]

            for q_idx, q in enumerate(ann["language_queries"]):
                start = q["clip_start_sec"]
                end   = q["clip_end_sec"]

                gt_lookup[(clip_uid, ann_uid, q_idx)] = [start, end]

print("Total GT queries:", len(gt_lookup))

# ==========================================================
# IoU FUNCTION
# ==========================================================
def temporal_iou(pred, gt):
    ps, pe = pred
    gs, ge = gt

    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)

    if union <= 0:
        return 0.0
    return inter / union

# ==========================================================
# METRIC STORAGE
# ==========================================================
ious_top1 = []
ious_top5_max = []

r1_03, r1_05 = 0, 0
r5_03, r5_05 = 0, 0

total = 0

# ==========================================================
# PROCESS PREDICTIONS
# ==========================================================
for item in pred_data["results"]:

    clip_uid = item["clip_uid"].split("_ann-")[0].replace("clip-", "")
    ann_uid = item["annotation_uid"]
    q_idx = item["query_idx"]

    key = (clip_uid, ann_uid, q_idx)

    if key not in gt_lookup:
        continue

    gt_segment = gt_lookup[key]
    preds = item["predicted_times"]

    # -----------------------------
    # TOP-1 IoU
    # -----------------------------
    iou_top1 = temporal_iou(preds[0], gt_segment)
    ious_top1.append(iou_top1)

    # -----------------------------
    # TOP-5 IoU (best of 5)
    # -----------------------------
    iou_list = [temporal_iou(p, gt_segment) for p in preds[:5]]
    best_iou = max(iou_list)
    ious_top5_max.append(best_iou)

    # -----------------------------
    # METRICS
    # -----------------------------
    if iou_top1 >= 0.3:
        r1_03 += 1
    if iou_top1 >= 0.5:
        r1_05 += 1

    if best_iou >= 0.3:
        r5_03 += 1
    if best_iou >= 0.5:
        r5_05 += 1

    total += 1

# ==========================================================
# FINAL METRICS
# ==========================================================
r1_03 /= total
r1_05 /= total
r5_03 /= total
r5_05 /= total

mean_r1 = (r1_03 + r1_05) / 2
miou = np.mean(ious_top1)

print("\n======================")
print(f"Total Samples: {total}")
print(f"mIoU: {miou:.4f}")
print(f"R@1 IoU=0.3: {r1_03:.4f}")
print(f"R@1 IoU=0.5: {r1_05:.4f}")
print(f"Mean R@1: {mean_r1:.4f}")
print(f"R@5 IoU=0.3: {r5_03:.4f}")
print(f"R@5 IoU=0.5: {r5_05:.4f}")
print("======================")

# ==========================================================
# PLOTS
# ==========================================================

# -----------------------------
# 1. IoU per query
# -----------------------------
plt.figure(figsize=(14,5))
plt.plot(ious_top1, linewidth=1)
plt.axhline(miou, linestyle='--', label=f"mIoU={miou:.4f}")
plt.title("Top-1 IoU per Query (Combined)")
plt.xlabel("Query Index")
plt.ylabel("IoU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "IoU_per_query_Combined.png")
plt.close()

# -----------------------------
# 2. IoU Distribution Histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(ious_top1, bins=30)
plt.title("IoU Distribution (Top-1) (Combined)")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "IoU_distribution_Combined.png")
plt.close()

# -----------------------------
# 3. Recall Bar Plot
# -----------------------------
labels = ["R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5"]
values = [r1_03, r1_05, r5_03, r5_05]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.title("Recall Metrics (Combined)")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(save_dir / "Recall_metrics_Combined.png")
plt.close()

# -----------------------------
# 4. Cumulative IoU Curve
# -----------------------------
sorted_ious = np.sort(ious_top1)
cum = np.arange(len(sorted_ious)) / len(sorted_ious)

plt.figure(figsize=(8,5))
plt.plot(sorted_ious, cum)
plt.title("Cumulative IoU Curve (Combined)")
plt.xlabel("IoU")
plt.ylabel("CDF")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "CDF_IoU_Combined.png")
plt.close()

print("\nAll plots saved in:", save_dir)
