import json
import copy
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
SOURCE_PRED_FILE = "/DATA/Ego4d/episodic-memory/NLQ/VSLNet/checkpoints/vslnet_nlq_official_v1_official_EgoVLP_903_bert/model/vslnet_27744_test_result.json"   # your 1st file
NLQ_ALL_FILE     = "/DATA/Ego4d/Ego4d_data/v2/annotations/nlq_all.json"             # your 3rd file
OUTPUT_DIR       = "/DATA/Ego4d/episodic-memory/NLQ/2D-TAN/Test_jsons"
FPS = 30  # Ego4D uses 30 fps

# ============================================================
# HELPERS
# ============================================================

def sec_to_frame(sec, fps=30):
    return int(round(sec * fps))

def round_sec(x):
    return round(float(x), 3)

def make_clip_uid(video_uid, ann_uid, q_idx, start_sec, end_sec):
    return (
        f"clip-{video_uid}"
        f"_ann-{ann_uid}"
        f"_q-{q_idx}"
        f"_s-{round_sec(start_sec)}"
        f"_e-{round_sec(end_sec)}"
    )

# ============================================================
# LOAD FILES
# ============================================================

with open(SOURCE_PRED_FILE, "r") as f:
    pred_data = json.load(f)

with open(NLQ_ALL_FILE, "r") as f:
    nlq_all = json.load(f)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# BUILD LOOKUP FROM nlq_all.json
# key = (clip_uid, annotation_uid)
# NOTE:
# user said:
# source.clip_uid matches target.video_uid and nlq_all.clip_uid
# so we use nlq_all clip_uid
# ============================================================

lookup = {}

for video in nlq_all["videos"]:
    for clip in video["clips"]:
        clip_uid = clip["clip_uid"]

        for ann in clip["annotations"]:
            ann_uid = ann["annotation_uid"]

            lookup[(clip_uid, ann_uid)] = {
                "video_uid": video["video_uid"],
                "split": video.get("split", "test"),
                "clip_meta": clip,
                "annotation": ann
            }

# ============================================================
# PREPARE 10 OUTPUT FILE STRUCTURES
# one file per predicted_times index
# ============================================================

outputs = []

for pred_idx in range(10):
    outputs.append({
        "version": "2.0",
        "date": "230106",
        "description": f"NLQ Query-Level Clips for 2D-TAN (prediction #{pred_idx})",
        "videos": []
    })

# keep map inside each output so same video_uid groups together
video_maps = [{} for _ in range(10)]

# ============================================================
# MAIN CONVERSION
# ============================================================

for item in pred_data["results"]:

    src_clip_uid = item["clip_uid"]
    ann_uid = item["annotation_uid"]
    q_idx = item["query_idx"]
    predicted_times = item["predicted_times"]

    key = (src_clip_uid, ann_uid)

    if key not in lookup:
        print(f"WARNING: Missing metadata for {key}")
        continue

    meta = lookup[key]

    video_uid = meta["video_uid"]
    clip_uid = meta["clip_meta"]["clip_uid"]
    split = meta["split"]
    orig_clip = meta["clip_meta"]
    orig_ann = meta["annotation"]

    # get corresponding language query
    lqs = orig_ann["language_queries"]

    if q_idx >= len(lqs):
        print(f"WARNING: query_idx {q_idx} out of range for {key}")
        continue

    lang_query = lqs[q_idx]

    # --------------------------------------------------------
    # create one entry in each of 10 output files
    # --------------------------------------------------------
    for pred_idx in range(min(10, len(predicted_times))):

        start_sec = round_sec(predicted_times[pred_idx][0])
        end_sec   = round_sec(predicted_times[pred_idx][1])

        start_frame = sec_to_frame(start_sec, FPS)
        end_frame   = sec_to_frame(end_sec, FPS)

        new_clip_uid = make_clip_uid(
            # video_uid,
            clip_uid,
            ann_uid,
            q_idx,
            start_sec,
            end_sec
        )

        # language query copy with updated temporal fields
        new_query = copy.deepcopy(lang_query)

        new_query["clip_start_sec"] = start_sec
        new_query["clip_end_sec"] = end_sec
        new_query["video_start_sec"] = start_sec
        new_query["video_end_sec"] = end_sec
        new_query["video_start_frame"] = start_frame
        new_query["video_end_frame"] = end_frame

        new_ann = {
            "annotation_uid": ann_uid,
            "language_queries": [new_query]
        }

        new_clip = {
            "clip_uid": new_clip_uid,
            "video_start_sec": start_sec,
            "video_end_sec": end_sec,
            "clip_start_sec": start_sec,
            "clip_end_sec": end_sec,
            "video_start_frame": start_frame,
            "video_end_frame": end_frame,
            "clip_start_frame": start_frame,
            "clip_end_frame": end_frame,
            "source_clip_uid": src_clip_uid,
            "annotations": [new_ann]
        }

        # ----------------------------------------------------
        # append under correct video in this output file
        # ----------------------------------------------------
        if video_uid not in video_maps[pred_idx]:
            new_video = {
                # "video_uid": video_uid,
                "video_uid": src_clip_uid,
                "clips": [],
                "split": split
            }
            outputs[pred_idx]["videos"].append(new_video)
            video_maps[pred_idx][video_uid] = new_video

        video_maps[pred_idx][video_uid]["clips"].append(new_clip)

# ============================================================
# SAVE 10 FILES
# ============================================================

for pred_idx in range(10):
    out_path = Path(OUTPUT_DIR) / f"nlq_prediction_index_{pred_idx}.json"

    with open(out_path, "w") as f:
        json.dump(outputs[pred_idx], f, indent=2)

    print(f"Saved: {out_path}")

print("Done.")
