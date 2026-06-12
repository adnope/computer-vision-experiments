# MAE Reproduction Experiments

Reproduction workflow for **Masked Autoencoders Are Scalable Vision Learners** using the official MAE codebase.

This project runs lightweight experiments for the Computer Vision course final semester report:

- Fine-tune MAE-pretrained ViT-B/16 on CIFAR-10.
- Train ViT-B/16 from scratch on CIFAR-10 as a baseline.
- Run MAE linear probing.
- Generate reconstruction demo images.
- Summarize results and plot accuracy/loss curves.

## Structure

```text
.
├── Dockerfile
├── checkpoints/
├── datasets/
│   ├── cifar10-raw/
│   ├── cifar10-imagefolder/
│   └── demo-images/
└── outputs/
    ├── run_experiments.sh
    ├── run_reconstruction_demo.py
    ├── summarize_and_plot.py
    ├── exp01_cifar10_mae_finetune/
    ├── exp02_cifar10_scratch/
    ├── exp03_reconstruction_demo/
    └── exp04_cifar10_mae_linprobe/
````

## Build

```bash
docker build -t mae-report:cuda .
```

## Run container

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/datasets:/workspace/datasets" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/outputs:/workspace/outputs" \
  mae-report:cuda bash
```

## Checkpoints

MAE pretrained ViT-Base checkpoint:

```bash
wget -nc -P /workspace/checkpoints \
  https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

MAE visualization ViT-Large checkpoint:

```bash
wget -nc -P /workspace/checkpoints \
  https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
```

## Run experiments

Inside the container:

```bash
cd /workspace/mae
bash /workspace/outputs/run_experiments.sh
```

This runs:

```text
exp01_cifar10_mae_finetune
exp02_cifar10_scratch
exp04_cifar10_mae_linprobe
```

## Run reconstruction demo

```bash
python /workspace/outputs/run_reconstruction_demo.py
```

Output:

```text
outputs/exp03_reconstruction_demo/mae_reconstruction_grid.png
```

## Summarize and plot

```bash
python /workspace/outputs/summarize_and_plot.py
```

Generated files:

```text
outputs/summary.csv
outputs/cifar10_acc1_curve.png
outputs/cifar10_test_loss_curve.png
```

## Acknowledgement

Official MAE repository:

```text
https://github.com/facebookresearch/mae
```
