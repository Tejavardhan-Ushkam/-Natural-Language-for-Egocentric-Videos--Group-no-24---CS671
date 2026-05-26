import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# FILE PATHS
# ==========================================================
pred_dir = "/DATA/Ego4d/episodic-memory/NLQ/2D-TAN/"
gt_path  = "/DATA/Ego4d/Ego4d_data/v2/annotations/nlq_all.json"

save_dir = Path("./analysis_plots")
save_dir.mkdir(exist_ok=True)

# ==========================================================
# LOAD GT FILE
# ==========================================================
with open(gt_path, "r") as f:
    gt_data = json.load(f)

# ==========================================================
# LOAD ALL 10 HEAD FILES
# ==========================================================
all_pred_data = []

for head in range(10):
    pred_path = f"{pred_dir}/nlq_2dtan_submission_Head{head}.json"

    with open(pred_path, "r") as f:
        pred_json = json.load(f)

    all_pred_data.append(pred_json)

print(f"Loaded {len(all_pred_data)} prediction files.")

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
# BUILD PER-HEAD LOOKUP
# ==========================================================
# head_predictions[head][key] = predicted_times
# ==========================================================
head_predictions = []

for head in range(10):
    pred_lookup = {}

    for item in all_pred_data[head]["results"]:

        clip_uid = item["clip_uid"].split("_ann-")[0].replace("clip-", "")
        ann_uid  = item["annotation_uid"]
        q_idx    = item["query_idx"]

        key = (clip_uid, ann_uid, q_idx)

        pred_lookup[key] = item["predicted_times"]

    head_predictions.append(pred_lookup)

print("Built prediction lookup for all heads.")

# ==========================================================
# METRIC STORAGE
# ==========================================================
ious_top1 = []       # max Top-1 IoU across heads
ious_top5 = []       # max Top-5 IoU across heads

r1_03, r1_05 = 0, 0
r5_03, r5_05 = 0, 0

total = 0

# ==========================================================
# PROCESS EACH QUERY
# ==========================================================
for key, gt_segment in gt_lookup.items():

    per_head_top1 = []
    per_head_top5 = []

    # ----------------------------------------
    # CHECK ALL HEADS
    # ----------------------------------------
    for head in range(10):

        if key not in head_predictions[head]:
            continue

        preds = head_predictions[head][key]

        # Top-1
        iou1 = temporal_iou(preds[0], gt_segment)
        per_head_top1.append(iou1)

        # Top-5
        iou5_list = [temporal_iou(p, gt_segment) for p in preds[:5]]
        best5 = max(iou5_list)
        per_head_top5.append(best5)

    # if no prediction exists
    if len(per_head_top1) == 0:
        continue

    # ----------------------------------------
    # TAKE MAX ACROSS HEADS
    # ----------------------------------------
    final_top1 = max(per_head_top1)
    final_top5 = max(per_head_top5)

    ious_top1.append(final_top1)
    ious_top5.append(final_top5)

    # ----------------------------------------
    # METRICS
    # ----------------------------------------
    if final_top1 >= 0.3:
        r1_03 += 1
    if final_top1 >= 0.5:
        r1_05 += 1

    if final_top5 >= 0.3:
        r5_03 += 1
    if final_top5 >= 0.5:
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

print("\n==============================")
print("MAX IoU ACROSS HEAD0 to HEAD9")
print("==============================")
print(f"Total Samples: {total}")
print(f"mIoU: {miou:.4f}")
print(f"R@1 IoU=0.3: {r1_03:.4f}")
print(f"R@1 IoU=0.5: {r1_05:.4f}")
print(f"Mean R@1: {mean_r1:.4f}")
print(f"R@5 IoU=0.3: {r5_03:.4f}")
print(f"R@5 IoU=0.5: {r5_05:.4f}")
print("==============================")

# ==========================================================
# PLOTS
# ==========================================================

# ----------------------------------------------------------
# 1. IoU per query
# ----------------------------------------------------------
plt.figure(figsize=(14,5))
plt.plot(ious_top1, linewidth=1)
plt.axhline(miou, linestyle='--', label=f"mIoU={miou:.4f}")
plt.title("Max Top-1 IoU per Query Across Head0-Head9")
plt.xlabel("Query Index")
plt.ylabel("IoU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "IoU_per_query_MaxAcrossHeads.png")
plt.close()

# ----------------------------------------------------------
# 2. Histogram
# ----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(ious_top1, bins=30)
plt.title("IoU Distribution (Max Across Heads)")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "IoU_distribution_MaxAcrossHeads.png")
plt.close()

# ----------------------------------------------------------
# 3. Recall Metrics
# ----------------------------------------------------------
labels = ["R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5"]
values = [r1_03, r1_05, r5_03, r5_05]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.title("Recall Metrics (Max Across Heads)")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(save_dir / "Recall_metrics_MaxAcrossHeads.png")
plt.close()

# ----------------------------------------------------------
# 4. CDF Curve
# ----------------------------------------------------------
sorted_ious = np.sort(ious_top1)
cum = np.arange(len(sorted_ious)) / len(sorted_ious)

plt.figure(figsize=(8,5))
plt.plot(sorted_ious, cum)
plt.title("CDF IoU Curve (Max Across Heads)")
plt.xlabel("IoU")
plt.ylabel("CDF")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "CDF_IoU_MaxAcrossHeads.png")
plt.close()

print("\nAll plots saved in:", save_dir)
