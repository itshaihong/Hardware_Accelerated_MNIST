import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5 import LeNet5

# Train LeNet-5 on CPU and save model
def train_lenet5():
    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)

    # Use CPU
    device = torch.device('cpu')

    # MNIST normalization as used classically
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST training and test sets via torchvision
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # Model, optimizer, loss
    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Training LeNet-5 on MNIST (CPU)...")
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        start_t = time.time()
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        elapsed = time.time() - start_t
        print(f"Epoch {epoch}, Average Loss: {total_loss/len(train_loader):.4f}, Time: {elapsed:.2f}s")

    # Evaluate on torchvision test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy (torchvision test): {acc:.2f}%")

    # Save the model
    model_path = os.path.join('weights', 'lenet5.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    train_lenet5()