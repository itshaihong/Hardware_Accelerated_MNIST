import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet5 import LeNet5

# Normalization used during training and inference
MEAN = 0.1307
STD = 0.3081


def get_data_loaders(batch_size: int, num_workers: int):
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0,1]
        transforms.Normalize((MEAN,), (STD,)),       # normalize
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST (CPU)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--weights_path", type=str, default="weights/lenet5.pth")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)

    device = torch.device("cpu")  # Phase 1 is CPU-only

    train_loader, test_loader = get_data_loaders(args.batch_size, args.num_workers)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print(f"Training on device: {device}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        elapsed = time.time() - start
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | Epoch Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), args.weights_path)
    print(f"Weights saved to {args.weights_path}")


if __name__ == "__main__":
    main()