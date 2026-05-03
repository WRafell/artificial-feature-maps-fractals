from tqdm import tqdm
import torch
import random
import numpy as np
import torch.nn.functional as F


def train_patch_classifier(
        backbone: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        learning_rate: float, 
        max_epoch: int,
        device: str):
    
    backbone.to(device)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(max_epoch):
        running_loss, running_corrects = 0., 0.
        total_inputs = 0
        backbone.train()
        for batch in tqdm(train_loader, total=len(train_loader), desc=f"{epoch+1}/{max_epoch}. Train", leave=False):
            inputs = batch[0].cuda()
            labels = batch[1].cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = backbone(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = outputs.data.max(1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds==labels).item()
            total_inputs += labels.size(0)
        train_loss = running_loss / total_inputs
        train_acc = running_corrects / total_inputs

        running_loss, running_corrects = 0., 0.
        total_inputs = 0
        backbone.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f"{epoch+1}/{max_epoch}. Val", leave=False):
                inputs = batch[0].cuda()
                labels = batch[1].cuda()
                outputs = backbone(inputs)
                loss = F.cross_entropy(outputs, labels)

                _, preds = outputs.data.max(1)
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds==labels).item()
                total_inputs += labels.size(0)

        val_loss = running_loss / total_inputs
        val_acc = running_corrects / total_inputs

        if epoch%10 == 0 or epoch+1==max_epoch:
            print(f"[{epoch}/{max_epoch}] train loss: {train_loss:.3f}, train acc: {train_acc:.3f}. val loss: {val_loss:.3f}, val acc: {val_acc:.3f}.")

    return backbone


def test_patch_classifier(patch_classifier: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: str):

    running_loss, running_corrects = 0., 0.
    total_inputs = 0
    patch_classifier.eval()
    patch_classifier.to(device)
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Test", leave=False):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = patch_classifier(inputs)
            loss = F.cross_entropy(outputs, labels)

            _, preds = outputs.data.max(1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds==labels).item()
            total_inputs += labels.size(0)
    test_loss = running_loss / total_inputs
    test_acc = running_corrects / total_inputs
    print(f"Loss: {test_loss:.3f}, Acc: {test_acc:.3f}.")


def set_manual_seed(random_seed: int):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)