import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define the LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2) # Input 32x32, Output 28x28 (with padding to match original LeNet-5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)              # Output 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)          # Output 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)              # Output 5x5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)        # Output 1x1 (after flattening)
        
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 120) 
        
        x = F.tanh(self.fc1(x))
        x = self.fc2(x) # Output layer (no activation for classification with CrossEntropyLoss)
        return x

# Data preparation (MNIST dataset)
transform = transforms.Compose([
    transforms.Resize((32, 32)), # LeNet-5 expects 32x32 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize based on MNIST stats
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model instantiation and training (simplified)
if __name__ == '__main__':
    model = LeNet5(num_classes=10)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simple training loop (for demonstration)
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Evaluation (simplified)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the LeNet-5 model on the 10000 test images: {100 * correct / total:.2f}%')