# CIFAR-10 Experiments

Systematic exploration of deep learning architectures on CIFAR-10, building and debugging models from first principles.

## Key Results

| Model | Test Accuracy | Key Learnings |
|-------|---------------|---------------|
| Linear | ~40% | Baseline. |
| CNN | ~88% | Impact of Dropout and Batch Normalization. |
| ResNet | **96%** | Residual connections, LR scheduling, and Mixup/CutMix. |
| ViT (Custom) | ~88% | Challenges of training Transformers on small datasets. |
| ViT (Pre-trained) | **98.5%** | Transfer learning effectiveness. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train a model using the configuration files:

```bash
# Train a ResNet
python train.py --config configs/default.yaml --model_config configs/resnet.yaml

# Train a ViT
python train.py --config configs/default.yaml --model_config configs/vit.yaml
```

## Project Structure

- `models/`: Implementations of Linear, CNN, ResNet, and ViT architectures.
- `notes/`: Research logs and experiment tracking.
- `configs/`: YAML configuration files.
- `train.py`: Main training loop.

## Experiments

I keep detailed logs of my experiments, debugging steps, and results in the `notes/` directory.
- [CNN Notes](notes/CNN%20Classifier%20Notes.md)
- [ResNet Notes](notes/ResNet%20Classifier%20Notes.md)
- [ViT Notes](notes/ViT%20Classifier%20Notes.md)
