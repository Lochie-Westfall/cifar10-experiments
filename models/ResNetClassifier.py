import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out

class ResNetClassifier(nn.Module):
    def __init__(self, epochs, num_classes=4, lr=1e-4, weight_decay=0, D=32, mixup_alpha=1.0):
        super().__init__()

        self.D = D
        self.num_classes = num_classes
        self.lr = lr
        self.mixup_alpha = mixup_alpha

        self.resblocks = nn.Sequential(
            ResidualBlock(3, D, stride=1), # 32x32
            ResidualBlock(D, D, stride=1), # 32x32

            ResidualBlock(D, D*2, stride=2), # 16x16 - downsampled by stride
            ResidualBlock(D*2, D*2, stride=1), # 16x16 

            ResidualBlock(D*2, D*4, stride=2), # 8x8 
            ResidualBlock(D*4, D*4, stride=1), # 8x8 
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 1x1

        self.linear_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(D * 4, num_classes),
        )

        self.optimizer = optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=10,
            threshold=0.001,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    def forward(self, x):
        x = self.resblocks(x)
        x = self.avgpool(x)
        x = self.linear_classifier(x)

        return x

    def mixup_data(self, x, y):
        if self.mixup_alpha > 0 and self.training:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def fit(self, imgs, labels):
        imgs, labels_a, labels_b, lam = self.mixup_data(imgs, labels)

        pred = self.forward(imgs)
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(pred, labels_b)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss