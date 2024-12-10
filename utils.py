import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score


def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    total = 0
    correct = 0

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

        # Сохраняем для расчёта метрик
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=(total_loss / total), acc=(correct / total))

    avg_loss = total_loss / total
    accuracy = correct / total

    # Превращаем списки тензоров в numpy-массивы
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Дополнительные метрики
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, balanced_acc

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