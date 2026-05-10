from PIL import Image, ImageFilter
from torchmetrics import AUROC
from copy import deepcopy
from typing import TextIO
from tqdm import tqdm
import torch
import random
import torchvision
import torchvision.transforms as T


CTRANSPATH_WEIGHTS_PATH = 'models/ctranspath.pth'


def load_patch_classifier(
    backbone_name: str,
    num_classes: int,
    weights_path: str | None = None,
) -> torch.nn.Module:
    """Build a patch classifier by name. Optionally load a trained checkpoint.

    Returns a model whose forward(x) -> logits over num_classes.
    """
    if backbone_name == 'resnet50':
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'vit_tiny':
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    elif backbone_name == 'ctranspath':
        from src.encoders.ctranspath import build_ctranspath_classifier
        model = build_ctranspath_classifier(
            num_classes=num_classes,
            weights_path=CTRANSPATH_WEIGHTS_PATH,
            freeze_backbone=True,
        )
    else:
        raise NotImplementedError(f"{backbone_name} not implemented")

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def load_patch_encoder(
    backbone_name: str,
    weights_path: str | None = None,
    num_classes_for_loading: int = 3,
) -> tuple[torch.nn.Module, int]:
    """Build a patch encoder (no classification head). Returns (model, feature_dim).

    For resnet50 / vit_tiny, the trained classifier is loaded then the head is
    stripped — `weights_path` should point at the trained classifier checkpoint.
    For ctranspath, returns the raw pretrained encoder (no extra weights loaded).
    """
    if backbone_name == 'resnet50':
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes_for_loading)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
        model.layer4 = torch.nn.Identity()
        model.fc = torch.nn.Identity()
        feature_dim = 1024
    elif backbone_name == 'vit_tiny':
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes_for_loading)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
        model.head = torch.nn.Identity()
        feature_dim = 192
    elif backbone_name == 'ctranspath':
        from src.encoders.ctranspath import build_ctranspath_encoder, CTRANSPATH_FEATURE_DIM
        model = build_ctranspath_encoder(CTRANSPATH_WEIGHTS_PATH)
        feature_dim = CTRANSPATH_FEATURE_DIM
    else:
        raise NotImplementedError(f"{backbone_name} not implemented")

    model.eval()
    return model, feature_dim


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Transforms:
    def __init__(self, resize_int: int):
        self.train_transform = T.Compose([
            T.RandomResizedCrop(resize_int, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.8
            ),
            GaussianBlur(p=0.3),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = T.Compose([
            T.Resize((resize_int, resize_int)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class PatchLevelDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.img_paths = [x[0] for x in data]
        self.labels = torch.tensor([x[1] for x in data])
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transforms(img)
        return img, label, img_path  
    

def train_patch_classifier(
        backbone: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        learning_rate: float, 
        num_epochs: int,
        device: str, 
        writer: TextIO,
        saving_dir: str) -> torch.nn.Module:
    
    backbone.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, backbone.parameters()),
        lr=learning_rate,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    
    lowest_val_loss = 1e3
    best_model = None
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(num_epochs):
        train_loss = 0.
        train_corrects = 0
        backbone.train()
        for batch in tqdm(train_loader, total=len(train_loader), desc=f"{epoch+1}/{num_epochs}. Train", leave=False):                
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = backbone(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = outputs.data.max(1)
            train_loss += loss.item()
            train_corrects += torch.sum(preds == labels).item()
        train_loss = train_loss / len(train_loader)
        train_acc = train_corrects / len(train_loader.dataset)

        val_loss = 0.
        val_corrects = 0
        backbone.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f"{epoch+1}/{num_epochs}. Val", leave=False):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = backbone(inputs)
                loss = loss_function(outputs, labels)
                _, preds = outputs.data.max(1)
                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels).item()
        val_loss = val_loss / len(val_loader)
        val_acc = val_corrects / len(val_loader.dataset)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = deepcopy(backbone)
            torch.save(backbone.state_dict(), saving_dir)

        if epoch%10 == 0 or epoch+1==num_epochs:
            print(f"[{epoch}/{num_epochs}] train loss: {train_loss:.3f}, train acc: {train_acc:.3f}. " \
                    f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}.")
        print(f"[{epoch}/{num_epochs}] train loss: {train_loss:.3f}, train acc: {train_acc:.3f}. " \
                    f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}.", file=writer)

    return best_model


def test_patch_classifier(
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader, 
        device: str, 
        num_classes: int,
        writer: TextIO) -> None:
    test_loss = 0.
    test_corrects = 0
    all_outputs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Evaluation", leave=False):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            all_labels.extend(labels.cpu().tolist())
            outputs = model(inputs)
            all_outputs.extend(outputs.cpu())
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            _, preds = outputs.data.max(1)
            test_loss += loss.item()
            test_corrects += torch.sum(preds == labels).item()
    test_loss = test_loss / len(test_loader)
    test_acc = test_corrects / len(test_loader.dataset)
    all_outputs = torch.stack(all_outputs)
    all_labels = torch.tensor(all_labels)
    auroc = AUROC(task="multiclass", num_classes=num_classes)(all_outputs, all_labels)
    print(f"Test - Loss: {test_loss:.4f}. Accuracy: {test_acc:.4f}. AUROC: {auroc:.4f}")
    print(f"Test - Loss: {test_loss:.4f}. Accuracy: {test_acc:.4f}. AUROC: {auroc:.4f}\n", file=writer)
    return






















