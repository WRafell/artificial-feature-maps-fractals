from src.utils_feature_maps import Transforms
from src.utils_patch_classifier import load_patch_encoder
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import random
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
    patch_classifier, feature_dim = load_patch_encoder(
        backbone_name=BACKBONE_NAME,
        weights_path=patch_model_dir,
        num_classes_for_loading=len(CLASSES),
    )
    print(f"Encoder feature dim: {feature_dim}")
    patch_classifier.cuda()

    if TRANSFORMS == 'baseline':
        transforms = Transforms().base_transforms
    else:
        raise ValueError(f"{TRANSFORMS} is not a valid transforms")

    # Generating latents
    bag_sizes = []
    for row in tqdm(df.itertuples(), total=df.shape[0], desc=f"Organ: {organ}. Slide per class: {slides_per_class}"):
        latent_dir = f"patch_latents/{TRANSFORMS}/{BACKBONE_NAME}/{organ}/{slides_per_class}/{row.subset}/{row.condition}"
        Path(latent_dir).mkdir(parents=True, exist_ok=True)
        patches = [str(p) for p in Path(row.patch_dir).iterdir()]
        bag = []
        for patch in patches:
            patch_name = Path(patch).stem
            with open(patch, 'rb') as f:
                img = Image.open(f).convert("RGB")    
            img = transforms(img).cuda()
            with torch.no_grad():
                latents = patch_classifier(img.unsqueeze(0))
            
            bag.append(latents.squeeze(0).cpu())

        bag = torch.stack(bag, dim=0)
        bag_sizes.append(bag.size(0))
        torch.save(bag, f"{latent_dir}/{row.slide_name}.pth" )
    
    print(f"Bag size mean: {np.mean(bag_sizes):.3f}")

    return


if __name__=='__main__':
    for organ in ORGANS:
        for slides_per_class in SLIDES_PER_CLASS:
            main(organ=organ, slides_per_class=slides_per_class)
