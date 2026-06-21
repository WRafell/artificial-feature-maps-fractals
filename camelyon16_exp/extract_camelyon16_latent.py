"""Extract per-patch embeddings on Camelyon16, grouped per slide.

Produces one .pth file per slide containing a (num_patches, feature_dim) tensor
in `local/patch_latents/baseline/{backbone}/camelyon16/{subset}/{slide_label}/`.

CSV columns:
    slide_name, img_path, subset, loc_x, loc_y, slide_label, anno_path, patch_label
"""
from src.utils_feature_maps import Transforms
from src.utils_patch_classifier import load_patch_encoder
from camelyon16_exp.utils_camelyon16 import load_camelyon16_df, get_split_dfs, SLIDES_PER_CLASS_TRAIN
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
import pandas as pd


BACKBONE_NAME = 'ctranspath'  # one of: resnet50, vit_tiny, ctranspath
TRANSFORMS = 'baseline'
DEVICE = 'cuda:0'
BATCH_SIZE = 256
NUM_WORKERS = 4
RANDOM_SEED = 2024


class CamelyonPatchDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: list[str], transforms):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with open(self.img_paths[idx], 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transforms(img)


def main():
    df = load_camelyon16_df()
    splits = get_split_dfs(df, n_per_class=SLIDES_PER_CLASS_TRAIN, seed=RANDOM_SEED)
    # Keep only the sampled train slides; val/test untouched
    df = pd.concat([splits['train'], splits['val'], splits['test']], ignore_index=True)
    print(
        f"Slides — train: {splits['train']['slide_name'].nunique()}, "
        f"val: {splits['val']['slide_name'].nunique()}, "
        f"test: {splits['test']['slide_name'].nunique()}"
    )

    # Patch model checkpoint trained by train_camelyon16_patch_classifier.py.
    # For pure-encoder backbones (e.g. raw ctranspath) we can skip head loading,
    # but we keep the same convention so all latents share supervision context.
    patch_model_path = f"models/patch_level/{BACKBONE_NAME}/camelyon16_patch_classifier.pt"
    if Path(patch_model_path).exists():
        weights_path = patch_model_path
    else:
        print(f"[warn] {patch_model_path} not found — using untrained head / pretrained encoder")
        weights_path = None

    encoder, feature_dim = load_patch_encoder(
        backbone_name=BACKBONE_NAME,
        weights_path=weights_path,
        num_classes_for_loading=2,
    )
    encoder = encoder.to(DEVICE)
    print(f"Backbone: {BACKBONE_NAME} | feature_dim: {feature_dim}")

    transforms = Transforms().base_transforms

    out_root = Path(f"local/patch_latents/{TRANSFORMS}/{BACKBONE_NAME}/camelyon16")
    bag_sizes: list[int] = []

    for slide_name, group in tqdm(df.groupby('slide_name'), desc='slides'):
        subset = group['subset'].iloc[0]
        slide_label = int(group['slide_label'].iloc[0])
        out_dir = out_root / subset / str(slide_label)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{slide_name}.pth"
        if out_file.exists():
            continue

        img_paths = group['img_path'].tolist()
        dataset = CamelyonPatchDataset(img_paths=img_paths, transforms=transforms)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        bag: list[torch.Tensor] = []
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                feats = encoder(imgs)
                bag.append(feats.cpu())
        bag_t = torch.cat(bag, dim=0)
        torch.save(bag_t, out_file)
        bag_sizes.append(bag_t.size(0))

    if bag_sizes:
        print(f"Bag size mean: {np.mean(bag_sizes):.1f} | min: {min(bag_sizes)} | max: {max(bag_sizes)}")


if __name__ == '__main__':
    main()
