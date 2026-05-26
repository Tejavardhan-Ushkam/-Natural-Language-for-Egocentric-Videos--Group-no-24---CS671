import torch, os, json

# Step 1: check a feature file shape
feat = torch.load("/DATA/Ego4d/Ego4d_features/official1/54ed5c08-0f1a-43c9-ae73-ace068fee1e9.pt")
print("Feature shape:", feat.shape)  # (T, 512)

# Step 2: get that clip's duration from annotations
with open("/DATA/Ego4d/Ego4d_data/v2/annotations/nlq_train_new.json") as f:
    data = json.load(f)

for video in data["videos"]:
    for clip in video["clips"]:
        if clip["clip_uid"] == "54ed5c08-0f1a-43c9-ae73-ace068fee1e9":
            duration = float(clip["video_end_sec"]) - float(clip["video_start_sec"])
            print("Clip duration:", duration, "seconds")
            print("Actual FPS:", feat.shape[0] / duration)