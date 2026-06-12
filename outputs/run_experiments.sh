#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mae

# Safety patch for old MAE code on modern PyTorch.
if grep -q "from torch._six import inf" util/misc.py; then
  sed -i 's/from torch._six import inf/from math import inf/' util/misc.py
fi

# ============================================================
# Experiment 1: MAE pretrained + full fine-tuning
# ============================================================

OUT=/workspace/outputs/exp01_cifar10_mae_finetune
mkdir -p "$OUT"

cat > "$OUT/command.sh" <<'CMD'
python main_finetune.py \
  --model vit_base_patch16 \
  --finetune /workspace/checkpoints/mae_pretrain_vit_base.pth \
  --data_path /workspace/datasets/cifar10-imagefolder \
  --nb_classes 10 \
  --input_size 224 \
  --epochs 30 \
  --warmup_epochs 3 \
  --batch_size 16 \
  --accum_iter 8 \
  --blr 5e-4 \
  --layer_decay 0.65 \
  --weight_decay 0.05 \
  --drop_path 0.1 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --reprob 0.25 \
  --num_workers 4 \
  --seed 0 \
  --output_dir /workspace/outputs/exp01_cifar10_mae_finetune \
  --log_dir /workspace/outputs/exp01_cifar10_mae_finetune/tensorboard
CMD

echo "============================================================"
echo "Running exp01: MAE pretrained + fine-tuning"
echo "============================================================"
bash "$OUT/command.sh" 2>&1 | tee "$OUT/stdout.log"


# ============================================================
# Experiment 2: ViT train from scratch
# ============================================================

OUT=/workspace/outputs/exp02_cifar10_scratch
mkdir -p "$OUT"

cat > "$OUT/command.sh" <<'CMD'
python main_finetune.py \
  --model vit_base_patch16 \
  --data_path /workspace/datasets/cifar10-imagefolder \
  --nb_classes 10 \
  --input_size 224 \
  --epochs 30 \
  --warmup_epochs 3 \
  --batch_size 16 \
  --accum_iter 8 \
  --blr 5e-4 \
  --layer_decay 0.65 \
  --weight_decay 0.05 \
  --drop_path 0.1 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --reprob 0.25 \
  --num_workers 4 \
  --seed 0 \
  --output_dir /workspace/outputs/exp02_cifar10_scratch \
  --log_dir /workspace/outputs/exp02_cifar10_scratch/tensorboard
CMD

echo "============================================================"
echo "Running exp02: ViT train from scratch"
echo "============================================================"
bash "$OUT/command.sh" 2>&1 | tee "$OUT/stdout.log"


# ============================================================
# Experiment 3: MAE linear probing
# ============================================================

OUT=/workspace/outputs/exp04_cifar10_mae_linprobe
mkdir -p "$OUT"

cat > "$OUT/command.sh" <<'CMD'
python main_linprobe.py \
  --model vit_base_patch16 \
  --cls_token \
  --finetune /workspace/checkpoints/mae_pretrain_vit_base.pth \
  --data_path /workspace/datasets/cifar10-imagefolder \
  --nb_classes 10 \
  --input_size 224 \
  --epochs 30 \
  --warmup_epochs 3 \
  --batch_size 64 \
  --accum_iter 2 \
  --blr 0.1 \
  --weight_decay 0.0 \
  --num_workers 4 \
  --seed 0 \
  --output_dir /workspace/outputs/exp04_cifar10_mae_linprobe \
  --log_dir /workspace/outputs/exp04_cifar10_mae_linprobe/tensorboard
CMD

echo "============================================================"
echo "Running exp04: MAE linear probing"
echo "============================================================"
bash "$OUT/command.sh" 2>&1 | tee "$OUT/stdout.log"


# ============================================================
# Summarize + plot
# ============================================================

echo "============================================================"
echo "Generating summary.csv and plots"
echo "============================================================"

python /workspace/outputs/summarize_and_plot.py

echo "============================================================"
echo "All experiments finished"
echo "Outputs:"
echo "  /workspace/outputs/summary.csv"
echo "  /workspace/outputs/cifar10_acc1_curve.png"
echo "  /workspace/outputs/cifar10_test_loss_curve.png"
echo "============================================================"
