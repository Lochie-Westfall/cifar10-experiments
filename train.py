import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

import numpy as np

from models.CNNClassifier import CNNClassifier
from models.LinearClassifier import LinearClassifier
from models.ResNetClassifier import ResNetClassifier
from models.ViTClassifier import VitClassifier
from models.ViT_B_16Classifier import ViT_B_16_Classifier

from plotting import plot_loss

from data import get_data



from utils.config import load_config, merge_configs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/default.yaml')
parser.add_argument('--model_config', type=str, default=None)  # Optional override
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--reset-lr', action='store_true', help='Reset learning rate when resuming (use config LR instead of checkpoint LR)')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Optionally merge with model-specific config
if args.model_config:
    model_config = load_config(args.model_config)
    config = merge_configs(config, model_config)

device_config = config['device']
if device_config == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_config)

model_name = config['model']['name']
model_cfg = config[model_name.lower()]
train_dataloader, test_dataloader = get_data(
    batch_size=model_cfg['batch_size'],
    num_workers=config['data']['num_workers'],
    augmentation=config['data']['augmentation'],
)

mixup_alpha = model_cfg['mixup_alpha']
mixup_fn = None
if mixup_alpha > 0:
    mixup_fn = v2.RandomChoice([
        v2.MixUp(alpha=mixup_alpha, num_classes=model_cfg['num_classes']),
        v2.CutMix(alpha=mixup_alpha, num_classes=model_cfg['num_classes']),
    ])

def load_model():
    model_name = config['model']['name']
    model_cfg = config[model_name.lower()]

    print(f"Training Model: {model_name}")

    if model_name.lower() == 'cnn':
        model = CNNClassifier(num_classes=model_cfg['num_classes'], lr=model_cfg['lr'], D=model_cfg['D'], weight_decay=model_cfg["weight_decay"])
    elif model_name.lower() == 'resnet':
        model = ResNetClassifier(num_classes=model_cfg['num_classes'], lr=model_cfg['lr'], D=model_cfg['D'], weight_decay=model_cfg['weight_decay'], epochs=model_cfg['epochs'], mixup_alpha=model_cfg.get('mixup_alpha', 1.0))
    elif model_name.lower() == 'linear':
        model = LinearClassifier(size=32, num_classes=model_cfg['num_classes'], D=model_cfg['hidden_dim'], lr=model_cfg['lr'])
    elif model_name.lower() == 'vit':
        model = VitClassifier(warmup=model_cfg['warmup'], image_size=32, n_classes=model_cfg['num_classes'], patch_size=model_cfg['patch_size'], n_blocks=model_cfg['n_blocks'], n_hidden=model_cfg['D'], n_heads=model_cfg['n_heads'], lr=model_cfg['lr'], weight_decay=model_cfg['weight_decay'], dropout=model_cfg['dropout'], epochs=model_cfg['epochs'])
    elif model_name.lower() == 'vit_b_16':
        model = ViT_B_16_Classifier(
            weights=model_cfg['weights'],
            freeze_backbone=model_cfg.get('freeze_backbone', False),
            lr=model_cfg['lr'],
            weight_decay=model_cfg['weight_decay'],
            num_classes=model_cfg['num_classes'],
            warmup=model_cfg.get('warmup', 10)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)

model = load_model()

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories
checkpoint_dir = config['paths']['checkpoints']
plots_dir = config['paths']['plots']
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

test_accuracies = []
train_accuracies = []

# Resume from checkpoint if specified
if args.resume:
    print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.reset_lr:
        print("Resetting learning rate to config value (not loading optimizer/scheduler state)")
    else:
        if 'optimizer_state_dict' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    test_accuracies = checkpoint.get('test_accuracies', [])
    test_accuracies = checkpoint.get('test_accuracies', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    timestamp = checkpoint.get('timestamp', timestamp)
    print(f"Loaded checkpoint with {len(test_accuracies)} previous epochs")

epochs = model_cfg['epochs']
save_interval = config['saving']['save_interval']
start_epoch = len(test_accuracies)  # Continue from where we left off if resuming

def save_checkpoint_and_plot(epoch_num, is_final=False):
    """Save model checkpoint and plot"""
    # Periodic saves overwrite the same file, final save is separate
    if is_final:
        checkpoint_path = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_final.pt"
        plot_path = f"{plots_dir}/{type(model).__name__}_{timestamp}_final.png"
    else:
        checkpoint_path = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_latest.pt"
        plot_path = f"{plots_dir}/{type(model).__name__}_{timestamp}_latest.png"

    # Save checkpoint with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'scheduler_state_dict': model.scheduler.state_dict(),
        'test_accuracies': test_accuracies,
        'train_accuracies': train_accuracies,
        'timestamp': timestamp,
        'epoch': epoch_num
    }, checkpoint_path)

    # Save plot
    import matplotlib.pyplot as plt
    plot_loss(test_accuracies, alpha=config['plotting']['ema_alpha'], train_losses=train_accuracies, hyperparams=model_cfg)
    plt.gcf().canvas.draw()  # Ensure the current figure is fully rendered
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')

    save_type = "final" if is_final else "latest"
    print(f"Saved {save_type} checkpoint and plot (epoch {epoch_num if epoch_num is not None else len(test_accuracies)})")

for epoch in range(epochs):
    current_epoch = start_epoch + epoch + 1

    model.eval()  # Disable dropout for evaluation
    with torch.no_grad():
        # Evaluate on test data
        test_accuracy = 0
        for x in test_dataloader:
            imgs, labels = x
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model.forward(imgs)

            pred_classes = pred.argmax(dim=1)
            correct = (pred_classes == labels).sum().item()
            test_accuracy += correct

        test_accuracy /= len(test_dataloader.dataset)
        test_accuracies.append(test_accuracy * 100)

        # Evaluate on train data
        train_accuracy = 0
        for x in train_dataloader:
            imgs, labels = x
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model.forward(imgs)

            pred_classes = pred.argmax(dim=1)
            correct = (pred_classes == labels).sum().item()
            train_accuracy += correct

        train_accuracy /= len(train_dataloader.dataset)
        train_accuracies.append(train_accuracy * 100)


        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"Epoch {current_epoch}: Train Acc: {train_accuracy*100:.2f}%, Test Acc: {test_accuracy*100:.2f}%, LR:  {current_lr:.6f}")
            
        plot_loss(test_accuracies, alpha=config['plotting']['ema_alpha'], train_losses=train_accuracies, hyperparams=model_cfg)

    # Train
    model.train()  # Enable dropout for training
    for x in train_dataloader:
        imgs, labels = x
        imgs, labels = imgs.to(device), labels.to(device)

        if mixup_fn is not None:
            imgs, labels = mixup_fn(imgs, labels)

        model.fit(imgs, labels)

    model.scheduler.step()

    # Periodic saving
    if save_interval > 0 and current_epoch % save_interval == 0:
        save_checkpoint_and_plot(current_epoch, is_final=False)

# Save final checkpoint and plot
final_epoch = start_epoch + epochs
save_checkpoint_and_plot(final_epoch, is_final=True)

