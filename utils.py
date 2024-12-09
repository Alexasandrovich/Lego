import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Обернём dataloader в tqdm
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Обновим описание прогресса
        loop.set_postfix(loss=(total_loss / total), acc=(correct / total))

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=(total_loss / total), acc=(correct / total))

    if total == 0:
        return total_loss, 0.0
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy