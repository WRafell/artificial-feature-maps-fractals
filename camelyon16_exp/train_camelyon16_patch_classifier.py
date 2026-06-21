"""Train a patch-level classifier on Camelyon16.

Binary task: 0 = normal patch, 1 = tumor patch (`patch_label` column).

Slides: 50 sampled per `slide_label` from the train subset. Val / test come
from the CSV's existing `subset` column (unchanged).
"""
from src.utils_patch_classifier import (
    Transforms,
    PatchLevelDataset,
    train_patch_classifier,
    test_patch_classifier,
    load_patch_classifier,
)
from camelyon16_exp.utils_camelyon16 import load_camelyon16_df, get_split_dfs, SLIDES_PER_CLASS_TRAIN
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import random
import torch
import numpy as np
import pandas as pd


RANDOM_SEED = 2024
CLASSES = ['normal', 'tumor']
BACKBONE_NAME = 'resnet50'  # one of: resnet50, vit_tiny, ctranspath
BATCH_SIZE = 256
RESIZE = 224
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'cuda:0'
PATCHES_PER_CLASS_TRAIN = 50_000  # cap per patch_label inside the 50+50 train slides


def balance_patch_labels(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    out = []
    for label in [0, 1]:
        sub = df[df['patch_label'] == label]
        if cap is not None and len(sub) > cap:
            sub = sub.sample(n=cap, random_state=seed)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_dir = Path("logs/patch_level")
    log_dir.mkdir(exist_ok=True, parents=True)
    writer_path = log_dir / f"{BACKBONE_NAME}_camelyon16.txt"

    with open(writer_path, 'a') as writer:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(timestamp, file=writer)
        print(f"Backbone: {BACKBONE_NAME}", file=writer)
        print(f"Train slides per class: {SLIDES_PER_CLASS_TRAIN}", file=writer)

        df = load_camelyon16_df()
        print(f"Total patches in CSV: {len(df):,}")
        splits = get_split_dfs(df, n_per_class=SLIDES_PER_CLASS_TRAIN, seed=RANDOM_SEED)

        # Cap patches within the sampled train slides for tractable training
        splits['train'] = balance_patch_labels(splits['train'], cap=PATCHES_PER_CLASS_TRAIN, seed=RANDOM_SEED)

        for k, v in splits.items():
            slides = v['slide_name'].nunique()
            label_counts = v['patch_label'].value_counts().to_dict()
            line = f"{k}: {len(v):,} patches | {slides} slides | patch_label counts: {label_counts}"
            print(line)
            print(line, file=writer)

        train_t = Transforms(resize_int=RESIZE).train_transform
        test_t = Transforms(resize_int=RESIZE).test_transform

        dataloaders = {}
        for subset, t in [('train', train_t), ('val', test_t), ('test', test_t)]:
            data = splits[subset][['img_path', 'patch_label']].values
            dataset = PatchLevelDataset(data=data, transforms=t)
            dataloaders[subset] = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=(subset == 'train'),
                num_workers=NUM_WORKERS,
            )

        _ = next(iter(dataloaders['train']))
        print("Data is loaded")

        model_dir = Path(f"models/patch_level/{BACKBONE_NAME}")
        model_dir.mkdir(exist_ok=True, parents=True)
        saving_dir = str(model_dir / "camelyon16_patch_classifier.pt")

        backbone = load_patch_classifier(
            backbone_name=BACKBONE_NAME,
            num_classes=len(CLASSES),
        )
        backbone = backbone.to(DEVICE)

        best_model = train_patch_classifier(
            backbone=backbone,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            writer=writer,
            saving_dir=saving_dir,
        )

        test_patch_classifier(
            model=best_model,
            test_loader=dataloaders['test'],
            device=DEVICE,
            writer=writer,
            num_classes=len(CLASSES),
        )


if __name__ == '__main__':
    main()
