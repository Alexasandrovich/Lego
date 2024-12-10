import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from dataset_loader import LegoDataset, get_augmentation_transforms
from model import LegoClassifierModel
from utils import train_one_epoch, evaluate
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = '/data/alex/LEGO_dataset/spilt/train'

    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)

    train_dataset = LegoDataset(train_dir, transform=get_augmentation_transforms(), downscale_width=1024)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    model = LegoClassifierModel(num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    epochs = 50
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        avg_loss, accuracy, precision, recall, f1, balanced_acc = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Train loss: {avg_loss:.4f}, acc: {accuracy:.4f}, precision: {precision:.4f}, "
              f"recall: {recall:.4f}, f1: {f1:.4f}, balanced_acc: {balanced_acc:.4f} ")
        torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()