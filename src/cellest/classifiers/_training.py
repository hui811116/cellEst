import torch
import tqdm

def train_classifier(classifier, train_loader, optimizer, criterion, device):
    """train the classifier for one epoch"""
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm.tqdm(train_loader, desc="Train", leave=False)
    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_acc = (predicted == labels).sum().item() / labels.size(0) * 100
        progress.set_postfix(batch_acc=f"{batch_acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def build_optimizer(train_cfg, model):
    """build the optimizer named in the train config"""
    if train_cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )


def build_scheduler(train_cfg, optimizer):
    """build the LR scheduler named in the train config, if any"""
    if train_cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_cfg.step_size, gamma=train_cfg.gamma
        )
    if train_cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    return None


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, best_epoch, best_val_auc):
    checkpoint = {
        "classifier_state_dict": model.classifier.state_dict() if hasattr(model, "classifier") else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
    }
    torch.save(checkpoint, path)

def _head_module(model):
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
        return model.backbone.fc
    raise ValueError("Cannot identify the trainable classifier head for checkpointing.")
