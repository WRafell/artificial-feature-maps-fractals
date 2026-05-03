import torch.utils
from torchmetrics.functional import accuracy, auroc
from typing import List, Tuple, Dict, TextIO
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn.functional as F

def train_slide_classifier(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        learning_rate: float,
        device: str,
        writer: TextIO) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """
    Train a slide classifier model.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        device: Device to run the training on ('cuda' or 'cpu').

    Returns:
        Tuple containing the best model and a dictionary of training metrics.
    """
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    lowest_val_loss = float('inf')
    metrics = {}
    metrics = {key: [] for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_model = deepcopy(model)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):

        # TRAINING
        model.train()
        running_loss, running_corrects = 0.0, 0
        total_inputs = len(train_loader.dataset)
        for batch in tqdm(train_loader, total=len(train_loader), desc=f"[{epoch}/{num_epochs}] Train", leave=False):
            inputs = batch[0].to(device).float()
            labels = batch[1].to(device).long()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * len(labels)
            running_corrects += torch.sum(preds==labels).double().item()
        train_loss = running_loss / total_inputs
        train_acc = running_corrects / total_inputs

        # VALIDATION
        model.eval()
        running_loss, running_corrects = 0.0, 0
        total_inputs = len(val_loader.dataset)
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f"[{epoch}/{num_epochs}] Val", leave=False):
                inputs = batch[0].to(device).float()
                labels = batch[1].to(device).long()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * len(labels)
                running_corrects += torch.sum(preds==labels).double().item()
        val_loss = running_loss / total_inputs
        val_acc = running_corrects / total_inputs

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = deepcopy(model)

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        if (epoch%5==0) or (epoch+1==num_epochs):
            print(
                f"e[{epoch+1}/{num_epochs}] "
                f"train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, "
                f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}")
        print(
            f"e[{epoch+1}/{num_epochs}] "
            f"train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, "
            f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}", file=writer)
    return best_model, metrics


def test_slide_classifier(
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader, 
        classes: List[str], 
        device: str) -> list:
    """
    Test slide-level classifier model.

    Args:
        model: The model to test.
        test_loader: DataLoader for test data.
        classes: List of class names.
        device: Device to run the testing on ('cuda' or 'cpu').

    Returns:
        Tuple containing accuracy and AUROC scores.
    """
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    slide_names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            inputs = batch[0].to(device).float()
            labels = batch[1].to(device).long()
            slide_names.extend(batch[2])
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.append(labels)
            all_preds.append(outputs.argmax(dim=1))
            all_probs.append(probs)

    # Concatenate all batches
    all_labels = torch.cat(all_labels).cpu()
    all_preds = torch.cat(all_preds).cpu()
    all_probs = torch.cat(all_probs).cpu()
    all_results = [slide_names, all_labels, all_preds, all_probs]

    # Calculate metrics
    num_classes = len(classes)
    acc = accuracy(all_preds, all_labels, task='multiclass', num_classes=num_classes).item()
    auc = auroc(all_probs, all_labels, task='multiclass', num_classes=num_classes).item()

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test AUROC: {auc:.4f}")
        
    return acc, auc, all_results









