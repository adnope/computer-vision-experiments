from pathlib import Path
import random
import sys

sys.path.insert(0, "/workspace/mae")

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import models_mae


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("/workspace/checkpoints/mae_visualize_vit_large.pth")
# DATA_ROOT = Path("/workspace/datasets/cifar10-imagefolder/val")
DATA_ROOT = Path("/workspace/datasets/demo-images")
OUT_DIR = Path("/workspace/outputs/exp03_reconstruction_demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().squeeze(0)
    x = x.permute(1, 2, 0).numpy()
    return x.clip(0, 1)


def main():
    model = models_mae.mae_vit_large_patch16()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print("load_state_dict:", msg)

    model.to(DEVICE)
    model.eval()

    image_paths = sorted(list(DATA_ROOT.glob("*.png")) + list(DATA_ROOT.glob("*.jpg")) + list(DATA_ROOT.glob("*.jpeg")))
    if not image_paths:
        raise RuntimeError(f"No images found under {DATA_ROOT}")

    random.seed(0)
    selected = random.sample(image_paths, k=min(6, len(image_paths)))

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))

    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)

    for row, path in enumerate(selected):
        x = load_image(path).to(DEVICE)

        with torch.no_grad():
            loss, pred, mask = model(x, mask_ratio=0.75)

        pred = model.unpatchify(pred)

        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
        mask = model.unpatchify(mask)

        masked = x * (1 - mask)
        reconstruction = pred.clamp(0, 1)
        pasted = x * (1 - mask) + reconstruction * mask

        images = [
            (x, "Original"),
            (masked, "Masked 75%"),
            (reconstruction, "MAE reconstruction"),
            (pasted, "Reconstruction pasted"),
        ]

        for col, (img, title) in enumerate(images):
            axes[row][col].imshow(to_numpy_img(img))
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(title)

        axes[row][0].set_ylabel(path.parent.name, rotation=0, labelpad=35, va="center")

    plt.tight_layout()
    output_path = OUT_DIR / "mae_reconstruction_grid.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
