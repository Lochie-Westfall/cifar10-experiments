import numpy as np
import matplotlib.pyplot as plt

import torch

def imshow(img):
    img = img.cpu() if img.is_cuda else img  # Move from GPU
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_loss(losses, alpha=0.1, train_losses=None, hyperparams=None):
    plt.clf()
    losses = torch.Tensor(losses)
    plt.plot(losses.to("cpu"), label='Test', alpha=0.6)

    if train_losses is not None:
        train_losses = torch.Tensor(train_losses)
        plt.plot(train_losses.to("cpu"), label='Train', alpha=0.6)

    # EMA
    if len(losses) > 1:
        ema = torch.zeros_like(losses)
        ema[0] = losses[0]
        for i in range(1, len(losses)):
            ema[i] = alpha * losses[i] + (1 - alpha) * ema[i-1]
        plt.plot(ema.numpy(), label='Test EMA', linestyle='--')

    if train_losses is not None and len(train_losses) > 1:
        train_ema = torch.zeros_like(train_losses)
        train_ema[0] = train_losses[0]
        for i in range(1, len(train_losses)):
            train_ema[i] = alpha * train_losses[i] + (1 - alpha) * train_ema[i-1]
        plt.plot(train_ema.numpy(), label='Train EMA', linestyle='--')

    plt.legend()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylim(0, 100)  # Always show 0-100% range

    # Add hyperparameters as text
    if hyperparams is not None:
        param_text = '\n'.join([f'{k}: {v}' for k, v in hyperparams.items()])
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.pause(0.001)  # pause a bit so that plots are updated


def imshow_with_labels(img, true_label, pred_labels, class_names):
      img = img.cpu() if img.is_cuda else img
      img = img / 2 + 0.5  # unnormalize
      npimg = img.numpy()

      plt.figure(figsize=(8, 6))
      plt.imshow(np.transpose(npimg, (1, 2, 0)))

      # Build title with ground truth
      title = f"True: {class_names[true_label]}\n"

      # Add predictions
      for model_name, pred_idx in pred_labels.items():
          correct = pred_idx == true_label
          color = '✓' if correct else '✗'
          title += f"{color} {model_name}: {class_names[pred_idx]}\n"

      plt.title(title, fontsize=12, loc='left')
      plt.axis('off')
      plt.tight_layout()  # Add this line
      plt.show()