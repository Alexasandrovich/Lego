import torch
from torch.utils.data import DataLoader
from dataset_loader import LegoDataset, get_augmentation_transforms
from model import LegoClassifierModel
from utils import train_one_epoch, evaluate
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = '/data/alex/LEGO_dataset/spilt/train'
    val_dir = '/data/alex/LEGO_dataset/spilt/val'
    test_dir = '/data/alex/LEGO_dataset/spilt/test'

    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)

    train_dataset = LegoDataset(train_dir, transform=get_augmentation_transforms())
    val_dataset = LegoDataset(val_dir, transform=get_augmentation_transforms())
    test_dataset = LegoDataset(test_dir, transform=get_augmentation_transforms())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = LegoClassifierModel(num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    epochs = 50
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()