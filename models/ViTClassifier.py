import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, image_size=32, patch_size=16, embedding_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size**2)/(patch_size**2)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class UniformMLP(nn.Module):
    def __init__(self, dims, depth):
        super().__init__()

        self.model = nn.Sequential()
        for i in range(depth):
            self.model.append(nn.Linear(dims, dims))
            if i < depth - 1:
                self.model.append(nn.GELU())

    def forward(self, x):
        return self.model(x)

class TransformerEncoder (nn.Module):
    def __init__(self, dims = 64, n_heads = 12, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dims)
        self.multihead_attention = torch.nn.MultiheadAttention(embed_dim=dims, num_heads=n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(dims)
        self.mlp = UniformMLP(dims=dims, depth=3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.multihead_attention.forward(self.n1(x), self.n1(x), self.n1(x))[0]) + x
        x = self.dropout(self.mlp.forward(self.n2(x))) + x

        return x


class VitClassifier(nn.Module):
    def __init__(self, warmup=10, n_classes=10, image_size=32, patch_size=16, n_blocks=12, n_hidden=768, n_heads=12, lr=1e-4, weight_decay=1e-4, dropout=0.1, epochs=300):
        super().__init__()

        self.embed = PatchEmbed(image_size=image_size, patch_size=patch_size, embedding_dim=n_hidden)

        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, n_hidden))

        self.transformers = nn.Sequential(
        )

        for i in range(n_blocks):
            self.transformers.append(TransformerEncoder(dims=n_hidden, n_heads=n_heads, dropout=dropout))

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_classes)
        )

        self.optimizer = optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)

        # Warmup + Cosine annealing
        warmup_epochs = warmup
        cosine_epochs = epochs - warmup_epochs

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6
        )

        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        x = self.embed.forward(x)
        x = x + self.pos_embedding
        x = self.transformers.forward(x)
        x = torch.mean(x, dim=1)
        x = self.classifier.forward(x)

        return x
    
    def fit(self, imgs, labels):
        pred = self.forward(imgs)
        loss = self.criterion(pred, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss