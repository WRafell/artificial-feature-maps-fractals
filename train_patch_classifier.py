from src.utils_patch_classifier import Transforms, PatchLevelDataset, train_patch_classifier, test_patch_classifier
from torch.utils.data import DataLoader
from pathlib import Path
from typing import TextIO
from datetime import datetime
import torch
import random
import torchvision
import numpy as np
import pandas as pd


RANDOM_SEED = 2024
CLASSES = ['D', 'M', 'N']
BACKBONE_NAME = 'ctranspath'
BATCH_SIZE =  256
RESIZE = 224
NUM_WORKERS = 4
NUM_EPOCHS = 50
PATIENCE = 3
ORGANS = ['stomach', 'colon']
LEARNING_RATE = 0.001
SLIDES_PER_CLASS = [200, 150, 100, 50, 25]
DEVICE = 'cuda:0'



def main(organ: str, slides_per_class: int, writer: TextIO):

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Organ: {organ}. Slides per class: {slides_per_class}") 
    print(f"Organ: {organ}. Slides per class: {slides_per_class}", file=writer) 

    csv_file = f"csv/{organ}_patch-level_train_{slides_per_class}.csv"
    print(f"csv_file: {csv_file}")

    df = pd.read_csv(csv_file)
    train_transforms = Transforms(resize_int=RESIZE).train_transform
    test_transforms = Transforms(resize_int=RESIZE).test_transform
    dataloaders = {}
    for subset in ['train', 'val', 'test']:
        subdf = df.query(f"subset=='{subset}'")
        data = subdf[['img_path', 'label']].values
        dataset = PatchLevelDataset(
            data=data,
            transforms=train_transforms if subset=='train' else test_transforms)
        dataloaders[subset] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=subset=='train', num_workers=4)
    _ = next(iter(dataloaders['train']))
    print("Data is loaded")

    # Training
    Path(f"models/patch_level/{BACKBONE_NAME}").mkdir(exist_ok=True, parents=True)
    model_dir=f"models/patch_level/{BACKBONE_NAME}/{organ}_patch_classifier_{slides_per_class}.pt"

    if BACKBONE_NAME == 'resnet50':
        backbone = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(CLASSES))
    elif BACKBONE_NAME == 'vit_tiny':
        import timm
        backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=len(CLASSES))
    elif BACKBONE_NAME == 'ctranspath':
        from src.utils_patch_classifier import load_patch_classifier
        backbone = load_patch_classifier(
            backbone_name='ctranspath',
            num_classes=len(CLASSES),
        )
    else:
        raise NotImplementedError(f"{BACKBONE_NAME} not implemented")
    backbone = backbone.to(DEVICE)

    best_model = train_patch_classifier(
        backbone=backbone,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        writer=writer,
        saving_dir=model_dir)
    
    # Testing
    test_patch_classifier(
        model=best_model,
        test_loader=dataloaders['test'],
        device=DEVICE,
        writer=writer,
        num_classes=len(CLASSES))

    return


if __name__=='__main__':
    for organ in ['stomach']:
        for slides_per_class in SLIDES_PER_CLASS:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            writer_file = f"logs/patch_level/{BACKBONE_NAME}_{organ}_{slides_per_class}.txt"
            with open(writer_file, 'a') as writer:
                print(timestamp, file=writer)
                main(organ=organ, slides_per_class=slides_per_class, writer=writer)
