"""Generate Class Activation Maps for Camelyon16.

For each slide:
    1. Read all (img_path, loc_x, loc_y) patches.
    2. Run patch classifier -> softmax over {normal, tumor}.
    3. Place non-normal scores at (loc_y, loc_x) on a 2D CAM canvas of shape
       (num_classes, LSIZE, CSIZE).
    4. Persist CAM as a pickled list in
       `local/feature_maps/{TRANSFORMS}/{BACKBONE}/camelyon16/{subset}/{slide_label}/`.
"""
from src.utils_feature_maps import Transforms, base_feature_map
from src.utils_patch_classifier import load_patch_classifier
from camelyon16_exp.utils_camelyon16 import load_camelyon16_df, get_split_dfs, SLIDES_PER_CLASS_TRAIN
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import pickle
import pandas as pd


BACKBONE_NAME = 'resnet50'  # one of: resnet50, vit_tiny, ctranspath
TRANSFORMS = 'baseline'
DEVICE = 'cuda:0'
BATCH_SIZE = 256
NUM_WORKERS = 4
RANDOM_SEED = 2024

NUM_CLASSES = 2     # {normal, tumor}
NORMAL_LABEL = 0
LSIZE = 256         # canvas height (matches main paper for slide-model reuse)
CSIZE = 256         # canvas width  (square for Camelyon WSIs)
SAVE_HEATMAP = True


class CamelyonPatchWithCoords(torch.utils.data.Dataset):
    """Returns (image_tensor, row, col) so coords travel with the batch."""
    def __init__(self, rows, transforms):
        # rows: list of tuples (img_path, loc_y, loc_x)
        self.rows = rows
        self.transforms = transforms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, row, col = self.rows[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transforms(img), row, col


def _fit_to_canvas(coord: int, max_coord: int, canvas_size: int) -> int:
    """Map a raw grid coord into [0, canvas_size-1]."""
    if max_coord <= 0:
        return 0
    return min(int(round(coord / max_coord * (canvas_size - 1))), canvas_size - 1)


def build_camelyon_feature_map(
    slide_name: str,
    slide_label: int,
    group_df: pd.DataFrame,
    patch_classifier: torch.nn.Module,
    transforms,
    out_file: Path,
    heatmap_file: Path | None,
) -> None:
    feature_map = base_feature_map(LSIZE, CSIZE, NUM_CLASSES)

    max_y = int(group_df['loc_y'].max())
    max_x = int(group_df['loc_x'].max())
    rows = []
    for r in group_df.itertuples():
        row = _fit_to_canvas(int(r.loc_y), max_y, LSIZE)
        col = _fit_to_canvas(int(r.loc_x), max_x, CSIZE)
        rows.append((r.img_path, row, col))

    dataset = CamelyonPatchWithCoords(rows=rows, transforms=transforms)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    patch_classifier.eval()
    patch_classifier.to(DEVICE)
    with torch.no_grad():
        for imgs, batch_rows, batch_cols in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            outputs = patch_classifier(imgs)
            scores = F.softmax(outputs, dim=1)
            preds = scores.argmax(dim=1)
            for i in range(imgs.size(0)):
                if preds[i].item() != NORMAL_LABEL:
                    feature_map[0, :, int(batch_rows[i]), int(batch_cols[i])] = scores[i].cpu()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(feature_map.tolist(), f)

    if heatmap_file is not None:
        heatmap_file.parent.mkdir(parents=True, exist_ok=True)
        save_image(feature_map, str(heatmap_file))


def main():
    df = load_camelyon16_df()
    splits = get_split_dfs(df, n_per_class=SLIDES_PER_CLASS_TRAIN, seed=RANDOM_SEED)
    df = pd.concat([splits['train'], splits['val'], splits['test']], ignore_index=True)
    print(
        f"Slides — train: {splits['train']['slide_name'].nunique()}, "
        f"val: {splits['val']['slide_name'].nunique()}, "
        f"test: {splits['test']['slide_name'].nunique()}"
    )

    patch_model_path = f"models/patch_level/{BACKBONE_NAME}/camelyon16_patch_classifier.pt"
    assert Path(patch_model_path).exists(), f"Patch model missing: {patch_model_path}"

    patch_classifier = load_patch_classifier(
        backbone_name=BACKBONE_NAME,
        num_classes=NUM_CLASSES,
        weights_path=patch_model_path,
    )

    if TRANSFORMS == 'baseline':
        transforms = Transforms().base_transforms
    else:
        raise ValueError(f"{TRANSFORMS} is not a valid transform set")

    fm_root = Path(f"local/feature_maps/{TRANSFORMS}/{BACKBONE_NAME}/camelyon16")
    heatmap_root = Path(f"local/heatmap/{TRANSFORMS}/{BACKBONE_NAME}/camelyon16")

    for slide_name, group in tqdm(df.groupby('slide_name'), desc='slides'):
        subset = group['subset'].iloc[0]
        slide_label = int(group['slide_label'].iloc[0])
        out_file = fm_root / subset / str(slide_label) / f"{slide_name}.txt"
        heatmap_file = (
            heatmap_root / subset / str(slide_label) / f"{slide_name}.png"
            if SAVE_HEATMAP
            else None
        )
        if out_file.exists():
            continue
        build_camelyon_feature_map(
            slide_name=slide_name,
            slide_label=slide_label,
            group_df=group,
            patch_classifier=patch_classifier,
            transforms=transforms,
            out_file=out_file,
            heatmap_file=heatmap_file,
        )


if __name__ == '__main__':
    main()
