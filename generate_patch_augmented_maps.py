from src.utils_feature_maps import Transforms, build_feature_map
from pathlib import Path
from tqdm import tqdm
import torch
import random
import torchvision
import numpy as np
import pandas as pd


RANDOM_SEED = 2024
BACKBONE_NAME = 'resnet50'
ORGANS = ['stomach', 'colon']
CLASSES = ['D', 'M', 'N']
LSIZE = 256
CSIZE = 128
SLIDES_PER_CLASS = [200, 150, 100, 50, 25][::-1]




def main(organ: str, slides_per_class: int):

    print(f"Organ: {organ}. Slide fraction: {slides_per_class}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(f"csv/{organ}_slide-level_train_{slides_per_class}.csv")
    ori_train_df = df.query("subset == 'train'")
    # ori_train_df = train_df.copy()
    print(f"Size of training: {ori_train_df.shape[0]}")
    train_df = None
    for class_ in CLASSES:
        subdf = ori_train_df.query(f"condition=='{class_}'")
        subdf = subdf.sample(n=1000, replace=True)
        train_df = subdf if train_df is None else pd.concat([train_df, subdf], axis=0)

    print(f"Size of training after augmentation: {train_df.shape[0]}")

    patch_model_dir = f"models/patch_level/{BACKBONE_NAME}/{organ}_patch_classifier_{slides_per_class}.pt"
    assert Path(patch_model_dir).exists()
    print(f"Patch classifier: {patch_model_dir}")
    patch_classifier = torchvision.models.resnet50()
    patch_classifier.fc = torch.nn.Linear(patch_classifier.fc.in_features, len(CLASSES))
    patch_classifier.load_state_dict(torch.load(patch_model_dir))

    transforms = Transforms().augment_transforms

    # Generating feature maps
    for idx, row in tqdm(enumerate(train_df.itertuples()), total=train_df.shape[0], desc=f"Organ: {organ}. Slide per class: {slides_per_class}"):
        feature_map_dir = f"feature_maps/patch_augmented/{BACKBONE_NAME}/{organ}/{slides_per_class}/{row.subset}/{row.condition}"
        Path(feature_map_dir).mkdir(parents=True, exist_ok=True)
        patches = [str(p) for p in Path(row.patch_dir).iterdir()]
        build_feature_map(
            slide_name=row.slide_name,
            slide_label=row.label,
            feature_map_name=f"{row.slide_name}_{idx}",
            use_patch_classifier=True,
            patch_classifier=patch_classifier,
            batch_size=512,
            num_features=len(CLASSES),
            transforms=transforms,
            patches=patches,
            feature_map_dir=feature_map_dir,
            lsize=LSIZE,
            csize=CSIZE,
            normal_label=2,
            save_heatmap=True
        )

    return


if __name__=='__main__':
    for organ in ORGANS:
        for slides_per_class in SLIDES_PER_CLASS:
            main(organ=organ, slides_per_class=slides_per_class)

