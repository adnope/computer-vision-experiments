import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_ROOT = Path("/workspace/outputs")

EXPERIMENTS = {
    "exp01_cifar10_mae_finetune": "MAE pretrained + fine-tuning",
    "exp02_cifar10_scratch": "ViT train from scratch",
}

def read_log(exp_dir: Path):
    log_path = exp_dir / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    records = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records

all_rows = []
curves = {}

for exp_name, display_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_ROOT / exp_name
    records = read_log(exp_dir)

    valid_records = [r for r in records if "test_acc1" in r]
    if not valid_records:
        raise RuntimeError(f"No test_acc1 found in {exp_dir / 'log.txt'}")

    best = max(valid_records, key=lambda r: r["test_acc1"])
    last = valid_records[-1]

    all_rows.append({
        "experiment": exp_name,
        "name": display_name,
        "best_epoch": best.get("epoch", ""),
        "best_acc1": best.get("test_acc1", ""),
        "best_acc5": best.get("test_acc5", ""),
        "best_test_loss": best.get("test_loss", ""),
        "last_epoch": last.get("epoch", ""),
        "last_acc1": last.get("test_acc1", ""),
        "last_test_loss": last.get("test_loss", ""),
    })

    curves[display_name] = {
        "epoch": [r["epoch"] + 1 for r in valid_records],
        "acc1": [r["test_acc1"] for r in valid_records],
        "test_loss": [r["test_loss"] for r in valid_records],
        "train_loss": [r.get("train_loss") for r in valid_records],
    }

summary_path = OUTPUT_ROOT / "summary.csv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment",
        "name",
        "best_epoch",
        "best_acc1",
        "best_acc5",
        "best_test_loss",
        "last_epoch",
        "last_acc1",
        "last_test_loss",
    ])
    writer.writeheader()
    writer.writerows(all_rows)

print(summary_path.read_text())

# Accuracy curve
plt.figure(figsize=(8, 5))
for name, curve in curves.items():
    plt.plot(curve["epoch"], curve["acc1"], marker="o", label=name)
plt.xlabel("Epoch")
plt.ylabel("Top-1 accuracy (%)")
plt.title("CIFAR-10 validation accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "cifar10_acc1_curve.png", dpi=200)
plt.close()

# Test loss curve
plt.figure(figsize=(8, 5))
for name, curve in curves.items():
    plt.plot(curve["epoch"], curve["test_loss"], marker="o", label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("CIFAR-10 validation loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "cifar10_test_loss_curve.png", dpi=200)
plt.close()

print("Saved:")
print(OUTPUT_ROOT / "summary.csv")
print(OUTPUT_ROOT / "cifar10_acc1_curve.png")
print(OUTPUT_ROOT / "cifar10_test_loss_curve.png")
