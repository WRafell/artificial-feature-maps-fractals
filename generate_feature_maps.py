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
SLIDES_PER_CLASS = [200, 150, 100, 50, 25]
TRANSFORMS = 'baseline'



def main(organ: str, slides_per_class: int):

    print(f"Organ: {organ}. Slide fraction: {slides_per_class}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(f"csv/{organ}_slide-level_train_{slides_per_class}.csv")
    patch_model_dir = f"models/patch_level/{BACKBONE_NAME}/{organ}_patch_classifier_{slides_per_class}.pt"
    assert Path(patch_model_dir).exists()
    print(f"Patch classifier: {patch_model_dir}")
    patch_classifier = torchvision.models.resnet50()
    patch_classifier.fc = torch.nn.Linear(patch_classifier.fc.in_features, len(CLASSES))
    patch_classifier.load_state_dict(torch.load(patch_model_dir))

    if TRANSFORMS == 'baseline':
        transforms = Transforms().base_transforms
    else:
        raise ValueError(f"{TRANSFORMS} is not a valid transforms")

    # Generating feature maps
    for row in tqdm(df.itertuples(), total=df.shape[0], desc=f"Organ: {organ}. Slide per class: {slides_per_class}"):
        feature_map_dir = f"feature_maps/{TRANSFORMS}/{BACKBONE_NAME}/{organ}/{slides_per_class}/{row.subset}/{row.condition}"
        Path(feature_map_dir).mkdir(parents=True, exist_ok=True)
        patches = [str(p) for p in Path(row.patch_dir).iterdir()]
        build_feature_map(
            slide_name=row.slide_name,
            slide_label=row.label,
            feature_map_name=row.slide_name,
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

