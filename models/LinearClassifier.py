import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, size, D=32, num_classes=4, lr=1e-4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * size * size, D),
            nn.Linear(D, D),
            nn.Linear(D, num_classes),
        )
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.flatten(1)
        return self.model(x)

    def fit(self, imgs, labels):
        pred = self.forward(imgs)

        loss = self.criterion(pred, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss