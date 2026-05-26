import os
import torch

class ParamUpdateTracker:
    def __init__(self, file_path="param_updates.txt"):
        self.file_path = file_path
        self.before_params = {}

        # Reset file
        with open(self.file_path, "w") as f:
            f.write("===== PARAMETER UPDATE LOG (FULL SORTED) =====\n\n")

    def capture_before(self, model):
        """Call at START of epoch"""
        self.before_params = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.before_params[name] = param.data.clone().detach()

    def log_epoch(self, model, epoch):
        """Call at END of epoch"""

        param_changes = []
        total_change = 0
        total_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            before = self.before_params.get(name, None)
            if before is None:
                continue

            after = param.data

            abs_change = (after - before).abs().mean().item()
            before_mean = before.abs().mean().item()

            percent_change = (abs_change / (before_mean + 1e-12)) * 100

            param_changes.append((name, percent_change, abs_change))

            total_change += abs_change
            total_params += 1

        avg_change = total_change / (total_params + 1e-12)

        # 🔥 SORT: highest → lowest change
        param_changes.sort(key=lambda x: x[1], reverse=True)

        # -------- WRITE LOG --------
        with open(self.file_path, "a") as f:
            f.write(f"\n========== Epoch {epoch} ==========\n")
            f.write(f"Avg Param Change: {avg_change:.6e}\n")

            # interpretation
            if avg_change < 1e-7:
                f.write("⚠️ VERY LOW CHANGE → model not learning\n")
            elif avg_change > 1e-2:
                f.write("⚠️ VERY HIGH CHANGE → unstable training\n")
            else:
                f.write("✅ Normal learning\n")

            f.write("\n📊 ALL PARAMETERS (sorted by % change):\n\n")

            dead_count = 0

            for name, pct, abs_c in param_changes:
                f.write(f"{name}\n")
                f.write(f"  % Change: {pct:.6f}%\n")
                f.write(f"  Abs Change: {abs_c:.6e}\n\n")

                if pct < 1e-6:
                    dead_count += 1

            # summary
            f.write("------ SUMMARY ------\n")
            f.write(f"Total Params: {len(param_changes)}\n")
            f.write(f"Dead Params (<1e-6%): {dead_count}\n")

            if dead_count > 0.3 * len(param_changes):
                f.write("⚠️ Many params not updating → gradient issue / LR low\n")

            f.write("\n")