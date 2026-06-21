"""Train the slide-level classifier on Camelyon16 with artificial CAM augmentation.

Fixed number of real slides (the 50+50 sampled train slides) and varying
amounts of artificial CAMs per class.

Real CAM directory:
    local/feature_maps/baseline/{BACKBONE}/camelyon16/{train,val,test}/{0,1}/*.txt
Artificial CAM directory:
    local/feature_maps/artificial_camelyon16/{ARTIFICIAL_TYPE}/{0,1}/*.txt
"""
from src.utils_slide_classifier import train_slide_classifier, test_slide_classifier
from src.utils_feature_maps import FeatureMapDataset
from torch.utils.data import DataLoader
from typing import TextIO
from pathlib import Path
import pickle
import random
import re
import torch
import torchvision
import numpy as np
import pandas as pd


class Camelyon16FeatureMapDataset(FeatureMapDataset):
    """Pads the 2-channel Camelyon16 CAM to 3 channels so the ImageNet-pretrained
    ResNet50 slide model and its 3-channel Normalize transform stay compatible.
    """

    @staticmethod
    def load_features(file: Path) -> torch.Tensor:
        with file.open('rb') as f:
            features = torch.tensor(pickle.load(f)[0])
        if features.size(0) == 2:
            pad = torch.zeros(1, features.size(1), features.size(2), dtype=features.dtype)
            features = torch.cat([features, pad], dim=0)
        return features


RANDOM_SEED = 2024
CLASSES = ['0', '1']  # 0 = normal, 1 = tumor (slide_label)
BACKBONE_NAME = 'resnet50'  # or 'resnet50', 'vit_tiny'
ARTIFICIAL_TYPE = 'fractals'
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_RUNS = 3
ARTIFICIAL_PER_CLASS = [0, 25, 50, 100, 150]
DEVICE = 'cuda:0'


def main(artificial_per_class: int, writer: TextIO):

    # --- Real CAMs (fixed set of 50 normal + 50 tumor train slides) ---
    real_cam_dir = f"local/feature_maps/baseline/{BACKBONE_NAME}/camelyon16"
    assert Path(real_cam_dir).exists(), f"Real CAM dir missing: {real_cam_dir}"
    print(f"Real CAM dir: {real_cam_dir}")
    print(f"Real CAM dir: {real_cam_dir}", file=writer)

    real_feature_maps = [
        (subset, label, str(file))
        for subset in ['train', 'val', 'test']
        for label, class_ in enumerate(CLASSES)
        for file in Path(f"{real_cam_dir}/{subset}/{class_}").iterdir()
        if file.suffix == '.txt'
    ]
    real_feature_maps = pd.DataFrame(real_feature_maps, columns=['subset', 'label', 'data_path'])

    # --- Artificial CAMs ---
    artificial_map_dir = f"local/feature_maps/artificial_camelyon16/{ARTIFICIAL_TYPE}"
    if artificial_per_class > 0:
        assert Path(artificial_map_dir).exists(), f"Artificial CAM dir missing: {artificial_map_dir}"
        print(f"Artificial CAM dir: {artificial_map_dir}")
        print(f"Artificial CAM dir: {artificial_map_dir}", file=writer)
        artificial_maps = [
            ('train', label, str(file))
            for label, class_ in enumerate(CLASSES)
            for file in Path(f"{artificial_map_dir}/{class_}").iterdir()
            if file.suffix == '.txt'
        ]
        artificial_maps = pd.DataFrame(artificial_maps, columns=['subset', 'label', 'data_path'])

    # --- Compose train pool: all real train + sampled artificial ---
    train_data = []
    for label, _ in enumerate(CLASSES):
        real_subdf = real_feature_maps.query(f"label=={label} and subset=='train'")
        real_list = real_subdf[['data_path', 'label']].values.tolist()
        train_data.extend(real_list)
        num_real = len(real_list)

        num_artificial = 0
        if artificial_per_class > 0:
            art_subdf = artificial_maps.query(f"label=={label}")
            art_list = art_subdf[['data_path', 'label']].values.tolist()
            if artificial_per_class < len(art_list):
                art_list = random.sample(art_list, artificial_per_class)
            train_data.extend(art_list)
            num_artificial = len(art_list)

        line = f"Label {label}: real={num_real}, artificial={num_artificial}"
        print(line)
        print(line, file=writer)

    train_set = Camelyon16FeatureMapDataset(data=train_data, is_train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_data = real_feature_maps.query("subset=='val'")[['data_path', 'label']].values.tolist()
    val_set = Camelyon16FeatureMapDataset(data=val_data, is_train=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_data = real_feature_maps.query("subset=='test'")[['data_path', 'label']].values.tolist()
    test_set = Camelyon16FeatureMapDataset(data=test_data, is_train=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Slide model ---
    model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    slide_model, train_metrics = train_slide_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        writer=writer,
    )

    test_acc, test_auc, test_values = test_slide_classifier(
        model=slide_model,
        test_loader=test_loader,
        classes=CLASSES,
        device=DEVICE,
    )
    return train_metrics, test_acc, test_auc, test_values


if __name__ == '__main__':

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    basedir = Path(f"logs/slide_level/camelyon16/artificial/{ARTIFICIAL_TYPE}")
    basedir.mkdir(parents=True, exist_ok=True)
    writer_file = basedir / f"{BACKBONE_NAME}.txt"

    # Parse existing log for already-completed `artificial_per_class` values.
    done_combos: set[int] = set()
    if writer_file.exists():
        pat = re.compile(r"Artificial maps per class:\s*(\d+)[\s\S]*?ACCURACY:")
        with open(writer_file) as f:
            log_text = f.read()
        for m in pat.finditer(log_text):
            done_combos.add(int(m.group(1)))
        print(f"[camelyon16] resuming: {len(done_combos)} configs already done.")

    with open(writer_file, 'a') as writer:
        print(f"Random seed: {RANDOM_SEED}\n Using {ARTIFICIAL_TYPE} as artificial map", file=writer)
        for artificial_per_class in ARTIFICIAL_PER_CLASS:
            if artificial_per_class in done_combos:
                print(f"[skip] camelyon16 artificial={artificial_per_class} — already done")
                continue

            print('=' * 20, file=writer)
            header = f"Camelyon16. Artificial maps per class: {artificial_per_class}"
            print(header)
            print(header, file=writer)

            test_acc_list, test_auc_list = [], []
            for num_run in range(NUM_RUNS):
                print(f"Number of run: {num_run}")
                print(f"Number of run: {num_run}", file=writer)
                _, test_acc, test_auc, _ = main(
                    artificial_per_class=artificial_per_class,
                    writer=writer,
                )
                test_acc_list.append(test_acc)
                test_auc_list.append(test_auc)

            acc_mean, acc_std = np.array(test_acc_list).mean(), np.array(test_acc_list).std()
            auc_mean, auc_std = np.array(test_auc_list).mean(), np.array(test_auc_list).std()
            summary = (
                f"camelyon16. {BACKBONE_NAME}. ACCURACY: {acc_mean:.4f} +/- {acc_std:.4f}. "
                f"AUC: {auc_mean:.4f} +/- {auc_std:.4f}"
            )
            print(summary)
            print(summary + "\n", file=writer)
