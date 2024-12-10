import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset_loader import LegoDataset, get_augmentation_transforms
from model import LegoClassifierModel
from utils import train_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Train a LEGO classifier model.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--downscale_width', type=int, default=1024,
                        help='Downscale width for contour detection preprocess')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = sorted(os.listdir(args.train_dir))
    num_classes = len(classes)

    train_dataset = LegoDataset(args.train_dir,
                                transform=get_augmentation_transforms(),
                                downscale_width=args.downscale_width)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    model = LegoClassifierModel(num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        avg_loss, accuracy, precision, recall, f1, balanced_acc = train_one_epoch(model, optimizer, train_loader,
                                                                                  device)
        print(f"Train loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Precision: {precision:.4f} "
              f"| Recall: {recall:.4f} | F1: {f1:.4f} | Balanced Acc: {balanced_acc:.4f}")

        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")


if __name__ == '__main__':
    main()