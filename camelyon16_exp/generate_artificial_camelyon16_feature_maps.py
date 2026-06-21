"""Generate Artificial Class Activation Maps for Camelyon16.

Camelyon16 is a binary task (normal / tumor) and each WSI contains a single
tissue piece, so each artificial CAM places one large fractal silhouette
centered on the canvas (vs. 2-4 slices for the gastric / colorectal datasets).

Output files (same folder convention as `generate_artificial_feature_maps.py`):
    local/feature_maps/artificial_camelyon16/fractals/{slide_label}/{slide_label}_{k}.txt
    local/heatmap/artificial_camelyon16/fractals/{slide_label}/{slide_label}_{k}.jpg
"""
from src.utils_artificial_maps import (
    create_julia_fractal,
    center_crop,
    swap_channels,
)
from torchvision.utils import save_image
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
import numpy as np


# Number of artificial CAMs per class
K = 1000

# Camelyon16 slide labels: 0 = normal, 1 = tumor
CLASSES = ['0', '1']
NORMAL_LABEL = 0
FAKE_TYPE = 'fractals'

# Final canvas size (must match `generate_camelyon16_feature_maps.py`)
LSIZE = 256
CSIZE = 256

# Larger generation canvas for crisper fractals before resize
GEN_HEIGHT = 768
GEN_WIDTH = 768

# Fractal tissue size as fraction of the generation canvas
TISSUE_FRACTION = 0.65


def create_camelyon_artificial_map(
    label: int,
    num_features: int,
    height: int,
    width: int,
    scale: float,
    radius: int = 2,
) -> np.ndarray:
    """Build one (height, width, num_features) artificial CAM for Camelyon16.

    Single fractal tissue is generated, then centered on the canvas with a
    small random spatial jitter. Confidence scores are sampled to mimic the
    output distribution of a real patch classifier.
    """
    is_normal = (label == NORMAL_LABEL)
    h = int(height * TISSUE_FRACTION)
    w = int(width * TISSUE_FRACTION)

    cell = create_julia_fractal(height=h, width=w, scale=scale, radius=radius)

    if np.random.rand() > 0.5:
        bounding = min(cell.shape)
        cell = center_crop(cell, bounding)
        h, w = cell.shape

    cell *= (cell > 0.3)
    base_cell = np.repeat(cell, num_features).reshape((h, w, num_features))

    feature_map = np.zeros((height, width, num_features))

    if is_normal:
        # Normal slides: scores stay near zero everywhere on the tissue
        base_feature = np.random.uniform(0, 0.2, (h, w, num_features))
        feature_cell = F.softmax(torch.tensor(base_feature), dim=2).numpy()
        sparsity_mask = np.random.uniform(0, 1, size=feature_cell.shape) > 0.9
        final_cell = (feature_cell * sparsity_mask) * base_cell
    else:
        # Tumor slides: boost the positive (tumor) channel on the tissue
        base_feature = np.random.uniform(-1, 0, (h, w, num_features))
        feature = np.random.uniform(0.8, 1, (h, w))
        base_feature[:, :, label] = feature
        feature_cell = F.softmax(torch.tensor(base_feature), dim=2).numpy()
        feature_cell = feature_cell * base_cell
        noise = np.random.normal(0, 0.1, size=feature_cell.shape)
        swap_cell = swap_channels(feature_cell)
        final_cell = torch.tensor(swap_cell + noise)
        final_cell = F.softmax(final_cell, dim=2).numpy() * base_cell

    # Center the tissue with small random jitter so the placement varies
    jitter_y = np.random.randint(-height // 10, height // 10 + 1)
    jitter_x = np.random.randint(-width // 10, width // 10 + 1)
    cy = (height - h) // 2 + jitter_y
    cx = (width - w) // 2 + jitter_x
    cy = int(np.clip(cy, 0, height - h))
    cx = int(np.clip(cx, 0, width - w))
    feature_map[cy:cy + h, cx:cx + w, :] = final_cell

    return feature_map


def main():
    for label, class_ in enumerate(CLASSES):
        out_dir = Path(f"local/feature_maps/artificial_camelyon16/{FAKE_TYPE}/{class_}")
        out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = Path(f"local/heatmap/artificial_camelyon16/{FAKE_TYPE}/{class_}")
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for k in tqdm(range(K), total=K, desc=f"class {class_}"):
            scale = np.random.uniform(0.4, 0.95)
            artificial_map = create_camelyon_artificial_map(
                label=label,
                num_features=len(CLASSES),
                height=GEN_HEIGHT,
                width=GEN_WIDTH,
                scale=scale,
            )
            artificial_map = resize(artificial_map, (LSIZE, CSIZE))
            tensor = torch.from_numpy(artificial_map).permute(2, 0, 1).unsqueeze(0)

            with open(out_dir / f"{class_}_{k}.txt", "wb") as f:
                pickle.dump(tensor.tolist(), f)

            # Pad to 3 channels for JPEG viz (zero blue channel)
            heatmap = torch.cat(
                [tensor[0], torch.zeros(1, LSIZE, CSIZE, dtype=tensor.dtype)],
                dim=0,
            )
            save_image(heatmap, str(heatmap_dir / f"{class_}_{k}.jpg"))


if __name__ == '__main__':
    main()
