import torch
import torch.nn.functional as F
import tqdm
from typing import Literal, overload


@overload
def evaluate_classifier(
    classifier, val_loader, criterion, device, return_predictions: Literal[True]
) -> tuple[float, float, torch.Tensor, torch.Tensor]: ...


@overload
def evaluate_classifier(
    classifier, val_loader, criterion, device, return_predictions: Literal[False] = False
) -> tuple[float, float]: ...


def evaluate_classifier(
    classifier, val_loader, criterion, device, return_predictions: bool = False
):
    """evaluate the classifier on the validation set"""
    classifier.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        progress = tqdm.tqdm(val_loader, desc="Eval", leave=False)
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_acc = (predicted == labels).sum().item() / labels.size(0) * 100
            progress.set_postfix(batch_acc=f"{batch_acc:.2f}%")
            if return_predictions:
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    if return_predictions:
        return epoch_loss, epoch_acc, torch.cat(all_outputs), torch.cat(all_labels)
    return epoch_loss, epoch_acc