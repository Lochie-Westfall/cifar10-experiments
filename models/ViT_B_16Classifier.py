import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

class ViT_B_16_Classifier(nn.Module):
    def __init__(self, num_classes=10, weights='DEFAULT', freeze_backbone=False, lr=3e-4, weight_decay=1e-4, warmup=10):
        super().__init__()

        self.vit = torchvision.models.vit_b_16(weights = weights)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

        # Freeze backbone if requested (only train the classification head)
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            # Unfreeze the new classification head
            for param in self.vit.heads.head.parameters():
                param.requires_grad = True

        self.optimizer = optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1, 1.0, warmup)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vit.forward(x)

    def fit (self, x, y):
        pred = self.forward(x)
        loss = self.criterion(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss