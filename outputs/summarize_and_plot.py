import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("/workspace/outputs")

EXPERIMENTS = {
    "exp01_cifar10_mae_finetune": {
        "name": "MAE pretrained + fine-tuning",
        "short_name": "MAE fine-tuning",
        "dataset": "CIFAR-10",
        "model": "ViT-B/16",
        "training_mode": "Fine-tune toàn bộ từ MAE checkpoint",
        "expected_epochs": 30,
        "include_in_curves": True,
    },
    "exp02_cifar10_scratch": {
        "name": "ViT train from scratch",
        "short_name": "Scratch",
        "dataset": "CIFAR-10",
        "model": "ViT-B/16",
        "training_mode": "Train toàn bộ từ random initialization",
        "expected_epochs": 30,
        "include_in_curves": True,
    },
    "exp04_cifar10_mae_linprobe": {
        "name": "MAE linear probing",
        "short_name": "Linear probing",
        "dataset": "CIFAR-10",
        "model": "ViT-B/16",
        "training_mode": "Freeze encoder, train linear head",
        "expected_epochs": 30,
        "include_in_curves": True,
    },
}

QUALITATIVE_EXPERIMENTS = {
    "exp03_mae_reconstruction_demo": {
        "name": "MAE reconstruction demo",
        "short_name": "Reconstruction",
        "dataset": "Ảnh minh họa",
        "model": "ViT-Large MAE visualization checkpoint",
        "training_mode": "Không huấn luyện, chỉ chạy reconstruction demo",
        "status": "qualitative",
        "notes": "Không có metric định lượng; dùng ảnh mae_reconstruction_grid.png trong báo cáo.",
    }
}


def read_jsonl_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(log_path.read_text(errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Warning: skip invalid JSON in {log_path}:{line_no}: {exc}")
            continue

        if isinstance(obj, dict):
            records.append(obj)

    return records


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def fmt_float(value: Any, digits: int = 4) -> str:
    x = as_float(value)
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    x = as_float(value)
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def latex_escape(s: Any) -> str:
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def valid_eval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if "epoch" not in record:
            continue
        if as_float(record.get("test_acc1")) is None:
            continue
        out.append(record)
    return out


def summarize_experiment(exp_name: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[float]] | None]:
    exp_dir = OUTPUT_ROOT / exp_name
    log_path = exp_dir / "log.txt"

    records = read_jsonl_log(log_path)
    eval_records = valid_eval_records(records)

    base_row = {
        "experiment": exp_name,
        "name": cfg["name"],
        "short_name": cfg["short_name"],
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "training_mode": cfg["training_mode"],
        "expected_epochs": cfg["expected_epochs"],
        "epochs_logged": len(eval_records),
        "status": "",
        "best_epoch_log": "",
        "best_epoch": "",
        "best_acc1": "",
        "best_acc5_at_best_acc1": "",
        "best_test_loss_at_best_acc1": "",
        "last_epoch_log": "",
        "last_epoch": "",
        "last_acc1": "",
        "last_acc5": "",
        "last_test_loss": "",
        "last_train_loss": "",
        "n_parameters": "",
        "notes": "",
    }

    if not log_path.exists():
        base_row["status"] = "missing_log"
        base_row["notes"] = f"Missing {log_path}"
        return base_row, None

    if not eval_records:
        base_row["status"] = "no_eval_records"
        base_row["notes"] = f"No records with test_acc1 in {log_path}"
        return base_row, None

    best = max(eval_records, key=lambda r: float(r["test_acc1"]))
    last = eval_records[-1]

    best_epoch_log = int(best["epoch"])
    last_epoch_log = int(last["epoch"])

    expected_epochs = int(cfg["expected_epochs"])
    status = "completed" if last_epoch_log + 1 >= expected_epochs else "partial"

    row = dict(base_row)
    row.update({
        "status": status,
        "best_epoch_log": best_epoch_log,
        "best_epoch": best_epoch_log + 1,
        "best_acc1": fmt_percent(best.get("test_acc1")),
        "best_acc5_at_best_acc1": fmt_percent(best.get("test_acc5")),
        "best_test_loss_at_best_acc1": fmt_float(best.get("test_loss")),
        "last_epoch_log": last_epoch_log,
        "last_epoch": last_epoch_log + 1,
        "last_acc1": fmt_percent(last.get("test_acc1")),
        "last_acc5": fmt_percent(last.get("test_acc5")),
        "last_test_loss": fmt_float(last.get("test_loss")),
        "last_train_loss": fmt_float(last.get("train_loss")),
        "n_parameters": last.get("n_parameters", ""),
    })

    curve = {
        "epoch": [int(r["epoch"]) + 1 for r in eval_records],
        "acc1": [float(r["test_acc1"]) for r in eval_records],
        "acc5": [float(r["test_acc5"]) for r in eval_records if as_float(r.get("test_acc5")) is not None],
        "test_loss": [float(r["test_loss"]) for r in eval_records if as_float(r.get("test_loss")) is not None],
        "train_loss": [
            float(r["train_loss"]) if as_float(r.get("train_loss")) is not None else None
            for r in eval_records
        ],
    }

    return row, curve


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_latex_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    quantitative_rows = [
        r for r in rows
        if r.get("status") in {"completed", "partial"} and r.get("best_acc1")
    ]

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{llcccc}",
        r"        \toprule",
        r"        Mã TN & Thí nghiệm & Best epoch & Best Top-1 Acc. & Top-5 tại best epoch & Loss tại best epoch \\",
        r"        \midrule",
    ]

    for r in quantitative_rows:
        lines.append(
            "        "
            + latex_escape(r["experiment"].split("_")[0])
            + " & "
            + latex_escape(r["name"])
            + " & "
            + latex_escape(r["best_epoch"])
            + " & "
            + latex_escape(r["best_acc1"]) + r"\%"
            + " & "
            + latex_escape(r["best_acc5_at_best_acc1"]) + r"\%"
            + " & "
            + latex_escape(r["best_test_loss_at_best_acc1"])
            + r" \\"
        )

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        r"    \caption{Kết quả định lượng của các thí nghiệm trên CIFAR-10}",
        r"    \label{tab:group_cifar10_results}",
        r"\end{table}",
        "",
    ])

    path.write_text("\n".join(lines))


def write_milestones_csv(path: Path, curves: dict[str, dict[str, list[float]]]) -> None:
    milestone_epochs = [1, 2, 3, 10, 20, 30]
    rows = []

    for exp_name, curve in curves.items():
        epoch_to_idx = {int(e): i for i, e in enumerate(curve["epoch"])}
        row = {"experiment": exp_name}

        for ep in milestone_epochs:
            idx = epoch_to_idx.get(ep)
            row[f"epoch_{ep}_acc1"] = "" if idx is None else f"{curve['acc1'][idx]:.2f}"

        rows.append(row)

    fieldnames = ["experiment"] + [f"epoch_{ep}_acc1" for ep in milestone_epochs]
    write_csv(path, rows, fieldnames)


def plot_acc1_curve(path: Path, curves: dict[str, dict[str, list[float]]], labels: dict[str, str]) -> None:
    if not curves:
        return

    plt.figure(figsize=(8.5, 5.2))
    for exp_name, curve in curves.items():
        plt.plot(
            curve["epoch"],
            curve["acc1"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=labels.get(exp_name, exp_name),
        )

    plt.xlabel("Epoch")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("CIFAR-10 validation accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_loss_curve(path: Path, curves: dict[str, dict[str, list[float]]], labels: dict[str, str]) -> None:
    if not curves:
        return

    plt.figure(figsize=(8.5, 5.2))
    for exp_name, curve in curves.items():
        if not curve["test_loss"]:
            continue
        epochs = curve["epoch"][:len(curve["test_loss"])]
        plt.plot(
            epochs,
            curve["test_loss"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=labels.get(exp_name, exp_name),
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("CIFAR-10 validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_best_acc_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    bar_rows = [
        r for r in rows
        if r.get("status") in {"completed", "partial"} and as_float(r.get("best_acc1")) is not None
    ]

    if not bar_rows:
        return

    names = [r["short_name"] for r in bar_rows]
    values = [float(r["best_acc1"]) for r in bar_rows]

    plt.figure(figsize=(8.5, 5.2))
    plt.bar(names, values)

    for i, value in enumerate(values):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center", va="bottom")

    plt.ylabel("Best Top-1 accuracy (%)")
    plt.title("Best CIFAR-10 validation accuracy")
    plt.ylim(0, max(values) * 1.18)
    plt.xticks(rotation=10, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, list[float]]] = {}
    labels: dict[str, str] = {}

    for exp_name, cfg in EXPERIMENTS.items():
        row, curve = summarize_experiment(exp_name, cfg)
        rows.append(row)
        labels[exp_name] = cfg["short_name"]

        if curve is not None and cfg.get("include_in_curves", True):
            curves[exp_name] = curve

    for exp_name, cfg in QUALITATIVE_EXPERIMENTS.items():
        rows.append({
            "experiment": exp_name,
            "name": cfg["name"],
            "short_name": cfg["short_name"],
            "dataset": cfg["dataset"],
            "model": cfg["model"],
            "training_mode": cfg["training_mode"],
            "expected_epochs": "",
            "epochs_logged": "",
            "status": cfg["status"],
            "best_epoch_log": "",
            "best_epoch": "",
            "best_acc1": "",
            "best_acc5_at_best_acc1": "",
            "best_test_loss_at_best_acc1": "",
            "last_epoch_log": "",
            "last_epoch": "",
            "last_acc1": "",
            "last_acc5": "",
            "last_test_loss": "",
            "last_train_loss": "",
            "n_parameters": "",
            "notes": cfg["notes"],
        })

    fieldnames = [
        "experiment",
        "status",
        "name",
        "short_name",
        "dataset",
        "model",
        "training_mode",
        "expected_epochs",
        "epochs_logged",
        "best_epoch_log",
        "best_epoch",
        "best_acc1",
        "best_acc5_at_best_acc1",
        "best_test_loss_at_best_acc1",
        "last_epoch_log",
        "last_epoch",
        "last_acc1",
        "last_acc5",
        "last_test_loss",
        "last_train_loss",
        "n_parameters",
        "notes",
    ]

    summary_path = OUTPUT_ROOT / "summary.csv"
    summary_report_path = OUTPUT_ROOT / "summary_report.csv"
    latex_table_path = OUTPUT_ROOT / "summary_table.tex"
    milestones_path = OUTPUT_ROOT / "milestones.csv"

    write_csv(summary_path, rows, fieldnames)
    write_csv(summary_report_path, rows, fieldnames)
    write_latex_summary_table(latex_table_path, rows)
    write_milestones_csv(milestones_path, curves)

    acc_curve_path = OUTPUT_ROOT / "cifar10_acc1_curve.png"
    loss_curve_path = OUTPUT_ROOT / "cifar10_test_loss_curve.png"
    best_bar_path = OUTPUT_ROOT / "cifar10_best_acc1_bar.png"

    plot_acc1_curve(acc_curve_path, curves, labels)
    plot_loss_curve(loss_curve_path, curves, labels)
    plot_best_acc_bar(best_bar_path, rows)

    print("Saved:")
    for p in [
        summary_path,
        summary_report_path,
        latex_table_path,
        milestones_path,
        best_bar_path,
        acc_curve_path,
        loss_curve_path,
    ]:
        print(f"  {p}")

    print()
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
