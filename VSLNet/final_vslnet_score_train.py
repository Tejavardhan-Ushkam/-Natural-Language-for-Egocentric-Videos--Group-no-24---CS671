import re
import matplotlib.pyplot as plt

log_file = "final_vslnet_score_train.txt"  # change if needed

# Storage
steps = []
data = {
    k: {"score": [], "start": [], "end": []}
    for k in range(1, 11)
}

with open(log_file, "r") as f:
    lines = f.readlines()

current_step = None
current_k = None

for line in lines:
    line = line.strip()

    # Match Epoch/Step
    match = re.match(r"Epoch \d+, Step (\d+)", line)
    if match:
        current_step = int(match.group(1))
        steps.append(current_step)

    # Match Top-k
    match = re.match(r"Top-(\d+) Results:", line)
    if match:
        current_k = int(match.group(1))

    # Score
    match = re.match(r"Score:\s*([0-9.]+)", line)
    if match and current_k:
        data[current_k]["score"].append(float(match.group(1)))

    # End Extra
    match = re.match(r"Avg End Extra:\s*([0-9.]+)", line)
    if match and current_k:
        data[current_k]["end"].append(float(match.group(1)))

    # Start Extra
    match = re.match(r"Avg Start Extra:\s*([0-9.]+)", line)
    if match and current_k:
        data[current_k]["start"].append(float(match.group(1)))

# ------------------ PLOTTING ------------------

def plot_metric(metric_key, title, ylabel):
    plt.figure(figsize=(8, 5))

    for k in range(1, 11):
        plt.plot(steps, data[k][metric_key], marker='o', label=f"Top-{k}")

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}_train.png")
    plt.show()

# 1. Score plot
plot_metric("score", "Score vs Step", "Score")

# 2. Start Extra plot
plot_metric("start", "Avg Start Extra vs Step", "Start Extra")

# 3. End Extra plot
plot_metric("end", "Avg End Extra vs Step", "End Extra")
