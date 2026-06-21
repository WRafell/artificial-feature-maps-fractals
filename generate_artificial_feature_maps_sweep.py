"""Generate artificial-CAM variants for the confidence-score sensitivity ablation.

For each variant we hold the fractal shape generation fixed and vary one knob of
the score design (positive-confidence floor, noise level, or score mode), then
write the CAMs to a separate `ARTIFICIAL_TYPE` directory so that
`train_artificial_maps.py` can be pointed at each one in turn.

Output layout (consumed by train_artificial_maps.py via ARTIFICIAL_TYPE):
    local/feature_maps/artificial/{variant_name}/{D,M,N}/{class}_{k}.txt

Variants generated:
    pos_low sweep : fractals_pos0.5 / 0.6 / 0.7 / 0.8 / 0.9      (noise 0.1, uniform)
    noise sweep   : fractals_noise0.0 / 0.05 / 0.1 / 0.2          (pos_low 0.8, uniform)
    score mode    : fractals_constant / fractals_random           (pos_low 0.8, noise 0.1)

`fractals_pos0.8` and `fractals_noise0.1` both equal the main configuration; we
generate one canonical copy named `fractals_baseline` to avoid duplication.
"""
from src.utils_artificial_maps import create_julia_fractal, creature_artificial_map
from torchvision.utils import save_image
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
import numpy as np


RANDOM_SEED = 2024
K = 200                      # CAMs per class per variant (>= max artificial_per_class used)
CLASSES = ['D', 'M', 'N']
LSIZE, CSIZE = 256, 128
SAVE_HEATMAP = False         # heatmaps not needed for the sweep; keep it light

# (variant_name, kwargs for creature_artificial_map)
VARIANTS = [
    # canonical / main configuration
    ("fractals_baseline",  dict(pos_low=0.8, noise_std=0.1, score_mode='uniform')),
    # positive-confidence floor sweep
    ("fractals_pos0.5",    dict(pos_low=0.5, noise_std=0.1, score_mode='uniform')),
    ("fractals_pos0.6",    dict(pos_low=0.6, noise_std=0.1, score_mode='uniform')),
    ("fractals_pos0.7",    dict(pos_low=0.7, noise_std=0.1, score_mode='uniform')),
    ("fractals_pos0.9",    dict(pos_low=0.9, noise_std=0.1, score_mode='uniform')),
    # noise-level sweep
    ("fractals_noise0.0",  dict(pos_low=0.8, noise_std=0.0,  score_mode='uniform')),
    ("fractals_noise0.05", dict(pos_low=0.8, noise_std=0.05, score_mode='uniform')),
    ("fractals_noise0.2",  dict(pos_low=0.8, noise_std=0.2,  score_mode='uniform')),
    # score-mode controls
    ("fractals_constant",  dict(pos_low=0.8, noise_std=0.1, score_mode='constant')),
    ("fractals_random",    dict(pos_low=0.8, noise_std=0.1, score_mode='random')),
]


def generate_variant(variant_name: str, kwargs: dict) -> None:
    for label, class_ in enumerate(CLASSES):
        basedir = Path(f"./local/feature_maps/artificial/{variant_name}/{class_}")
        basedir.mkdir(parents=True, exist_ok=True)
        for k in tqdm(range(K), total=K, desc=f"{variant_name}/{class_}", leave=False):
            num_slices = np.random.choice([2, 3, 4], p=[0.15, 0.7, 0.15])
            scale = np.random.uniform(0.4, 0.95)
            artificial_map, _ = creature_artificial_map(
                create_fake_cell=create_julia_fractal,
                label=label,
                num_features=len(CLASSES),
                height=1280,
                width=640,
                num_slices=num_slices,
                scale=scale,
                is_normal=label == 2,
                **kwargs,
            )
            artificial_map = resize(artificial_map, (LSIZE, CSIZE))
            tensor = torch.from_numpy(artificial_map).permute(2, 0, 1).unsqueeze(0)
            with open(basedir / f"{class_}_{k}.txt", "wb") as f:
                pickle.dump(tensor.tolist(), f)

            if SAVE_HEATMAP:
                heatmap_dir = Path(str(basedir).replace('feature_maps', 'heatmap'))
                heatmap_dir.mkdir(parents=True, exist_ok=True)
                save_image(tensor[0], str(heatmap_dir / f"{class_}_{k}.jpg"))


def main() -> None:
    np.random.seed(RANDOM_SEED)
    for variant_name, kwargs in VARIANTS:
        print(f"=== {variant_name}  {kwargs} ===")
        generate_variant(variant_name, kwargs)


if __name__ == "__main__":
    main()
