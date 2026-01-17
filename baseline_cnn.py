import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Baseline CNN for underwater image enhancement.
    Three convolutional layers with ReLU activations.
    """
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)
