from src.utils_artificial_maps import create_julia_fractal, creature_artificial_map
from torchvision.utils import save_image
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
import numpy as np


K = 1000
CLASSES = ['D', 'M', 'N']
FAKE_TYPE = 'fractals'
LSIZE = 256
CSIZE = 128


def main():
    for label, class_ in enumerate(CLASSES):
        basedir = f"feature_maps/artificial/{FAKE_TYPE}/{class_}"
        Path(basedir).mkdir(exist_ok=True, parents=True)
        for k in tqdm(range(K), total=K, desc=class_):
            num_slices = np.random.choice([2,3,4], p=[0.15, 0.7, 0.15])
            scale = np.random.uniform(0.4, 0.95)
            artificial_map, _ = creature_artificial_map(
                create_fake_cell=create_julia_fractal, 
                label=label, 
                num_features=3, 
                height=1280, 
                width=640, 
                num_slices=num_slices, 
                scale=scale,
                is_normal=label==2)
            artificial_map = resize(artificial_map, (LSIZE, CSIZE))
            artificial_map = torch.from_numpy(artificial_map).permute(2,0,1).unsqueeze(0)
            with open(f"{basedir}/{class_}_{k}.txt", "wb") as f:
                pickle.dump(artificial_map.tolist(), f)

            heatmap_dir = basedir.replace('feature_maps', 'heatmap')
            Path(heatmap_dir).mkdir(exist_ok=True, parents=True)
            save_image(artificial_map[0], f"{heatmap_dir}/{class_}_{k}.jpg")

    return


if __name__=='__main__':
    main()
