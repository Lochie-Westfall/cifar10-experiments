import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1) -> None:
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2),
            )

    def forward(self, x):
        return self.model(x)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10, lr=1e-4, weight_decay=1e-4, D=32):
        super().__init__()
        self.model = nn.Sequential(
                CNNBlock(in_channels=3, out_channels=D),
                CNNBlock(in_channels=D, out_channels=D * 4),
                nn.Flatten(),
                nn.Linear(D * 16 * 16, D * 16),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(D * 16, D),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(D, num_classes)
            )
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def fit(self, imgs, labels):
        pred = self.forward(imgs)

        loss = self.criterion(pred, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss