import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture adapted for MNIST (1x28x28 input).
    Conv→Pool→Conv→Pool→FC→FC→FC with ReLU activations.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv1: 1 → 6, kernel 5x5, padding 2 to keep 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Conv2: 6 → 16, kernel 5x5, no padding -> 10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Fully connected layers
        # After Pool2: 16x5x5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Initialize weights (optional but typical)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Conv1 + ReLU -> Pool1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 6x14x14

        # Conv2 + ReLU -> Pool2
        x = F.relu(self.conv2(x))                     # 16x10x10
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 16x5x5

        # Flatten
        x = torch.flatten(x, 1)                       # N x 400

        # FC layers with ReLU
        x = F.relu(self.fc1(x))                       # 120
        x = F.relu(self.fc2(x))                       # 84
        x = self.fc3(x)                               # 10 logits
        return x