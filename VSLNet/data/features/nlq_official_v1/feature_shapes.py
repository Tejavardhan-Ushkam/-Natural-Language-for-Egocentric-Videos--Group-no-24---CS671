import os
import torch
import json
feature_dir = "/DATA/Ego4d/episodic-memory/NLQ/VSLNet/data/features/nlq_official_v1/official_EgoVLP"
save_dir = "/DATA/Ego4d/episodic-memory/NLQ/VSLNet/data/features/nlq_official_v1/official_EgoVLP"

feature_shapes = {}

for file in os.listdir(feature_dir):
    if file.endswith(".pt"):
        path = os.path.join(feature_dir, file)
        feat = torch.load(path)

        # assume shape = (T, D)
        feature_shapes[file.replace(".pt", "")] = feat.shape[0]

# save json
with open(os.path.join(save_dir, "feature_shapes.json"), "w") as f:
    json.dump(feature_shapes, f, indent=2)

print("Done!")
