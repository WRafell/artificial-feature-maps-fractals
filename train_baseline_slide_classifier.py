from src.utils_slide_classifier import train_slide_classifier, test_slide_classifier
from src.utils_feature_maps import FeatureMapDataset
from torch.utils.data import DataLoader
from datetime import datetime
from typing import TextIO
from pathlib import Path
import random
import torch
import torchvision
import numpy as np
import pandas as pd


RANDOM_SEED = 2024
CLASSES = ['D', 'M', 'N']
BACKBONE_NAME = 'vit_tiny'
BATCH_SIZE =  64
NUM_WORKERS = 4
NUM_EPOCHS = 50
ORGANS = ['stomach', 'colon']
LEARNING_RATE = 0.001
NUM_RUNS = 5
SLIDES_PER_CLASS = [150, 100, 50, 25]
DEVICE = 'cuda:0'


def main(organ: str, slides_per_class: int, writer: TextIO):

    # LOADING DATA
    feature_map_dir = f"feature_maps/baseline/{BACKBONE_NAME}/{organ}/{slides_per_class}"
    assert Path(feature_map_dir).exists(), "Feature maps directory does not exist"
    print(f"Feature maps directory: {feature_map_dir}") 
    print(f"Feature maps directory: {feature_map_dir}", file=writer) 

    feature_maps = [
        (subset, label, str(file))
        for subset in ['train', 'val', 'test']
        for label, class_ in enumerate(sorted(CLASSES))
        for file in Path(f"{feature_map_dir}/{subset}/{class_}").iterdir() if file.suffix == '.txt'
    ]
    feature_maps = pd.DataFrame(feature_maps, columns=['subset', 'label', 'data_path'])

    train_data = feature_maps.query("subset=='train'")[['data_path', 'label']].values.tolist()
    train_set = FeatureMapDataset(data=train_data, is_train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_data = feature_maps.query("subset=='val'")[['data_path', 'label']].values.tolist()
    val_set = FeatureMapDataset(data=val_data, is_train=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_data = feature_maps.query("subset=='test'")[['data_path', 'label']].values.tolist()
    test_set = FeatureMapDataset(data=test_data, is_train=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    # TRAINING
    model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    slide_model, train_metrics = train_slide_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        writer=writer
    )

    # TESTING
    test_acc, test_auc, test_values = test_slide_classifier(
        model=slide_model,
        test_loader=test_loader,
        classes=CLASSES,
        device=DEVICE
    )

    return train_metrics, test_acc, test_auc, test_values


if __name__=='__main__':

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for organ in ORGANS:
        basedir = f"logs/slide_level/{organ}/baseline/"
        Path(basedir).mkdir(exist_ok=True, parents=True)
        writer_file = f"{basedir}/{BACKBONE_NAME}.txt"
        train_results = {}
        test_results = {}
        with open(writer_file, 'w') as writer:  
            print(f"Random seed: {RANDOM_SEED}", file=writer)      
            for slides_per_class in SLIDES_PER_CLASS:
                print('='*20, file=writer)
                print(f"Organ: {organ}. Slides per class: {slides_per_class}") 
                print(f"Organ: {organ}. Slides per class: {slides_per_class}", file=writer) 
                train_results[slides_per_class] = {}
                test_results[slides_per_class] = {}
                test_acc_list = []
                test_auc_list = []
                for num_run in range(NUM_RUNS):
                    print(f"Number of run: {num_run}")
                    print(f"Number of run: {num_run}", file=writer)
                    train_metrics, test_acc, test_auc, test_values = main(
                        organ=organ, 
                        slides_per_class=slides_per_class,
                        writer=writer)
                    train_results[slides_per_class][num_run] = train_metrics
                    test_acc_list.append(test_acc)
                    test_auc_list.append(test_auc)
                test_results[slides_per_class]['acc'] = test_acc_list
                test_results[slides_per_class]['auc'] = test_auc_list

                acc_mean = np.array(test_acc_list).mean()
                acc_std = np.array(test_acc_list).std()
                auc_mean = np.array(test_auc_list).mean()
                auc_std = np.array(test_auc_list).std()

                print(f"{organ}. {BACKBONE_NAME}. ACCURACY: {acc_mean:.4f} +/- {acc_std:.4f}. AUC: {auc_mean:.4f} +/- {auc_std:.4f}")
                print(f"{organ}. {BACKBONE_NAME}. ACCURACY: {acc_mean:.4f} +/- {acc_std:.4f}. AUC: {auc_mean:.4f} +/- {auc_std:.4f}\n", file=writer)
                print()




