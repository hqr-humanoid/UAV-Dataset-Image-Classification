# UAV Dataset Image Classification

This project implements a UAV image classification pipeline using PyTorch and ResNet18. The code trains a neural network to classify UAV and background images, visualize predictions, and manage model checkpoints.

## Project Structure

```
|-- checkpoints/          # Folder to save model checkpoints
|-- dataset1/             # UAV dataset directory
|   |-- train/            # Training images
|   |-- val/              # Validation images
|   |-- test/             # Test images
|-- visualize/            # Folder to save visualization of predictions
|-- main.py               # Main script to train, evaluate, and visualize
|-- README.md             # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Torchvision
- PIL (Pillow)
- Numpy
- tqdm
- Matplotlib

Install dependencies:

```bash
pip install torch torchvision pillow numpy tqdm matplotlib
```

## Dataset Structure

The UAV dataset is divided into three parts:

- `train/` - Training images (UAV and background images)
- `val/` - Validation images
- `test/` - Test images

Each subdirectory contains:

- `UAV/` - UAV images (label 1)
- `background/` - Background images (label 0)

Example structure:

```
dataset1/
|-- train/
|   |-- UAV/
|   |-- background/
|-- val/
|   |-- UAV/
|   |-- background/
|-- test/
|   |-- UAV/
|   |-- background/
```

## Key Components

### 1. ImageDataset Class

Custom PyTorch dataset class to handle UAV and background images.

```python
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
```

- `uav_dir`: Path to UAV images
- `background_dir`: Path to background images
- `transform`: Image transformations (augmentation, resizing, etc.)

### 2. Training and Evaluation

- `train_model`: Trains the model for a specified number of epochs, saves checkpoints, and tracks performance.
- `evaluate_model`: Evaluates the model on test data and reports accuracy.
- `visualize_predictions`: Visualizes model predictions by displaying images and overlaying true and predicted labels.

### 3. Model Training

The model uses a pretrained ResNet18:

```python
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
```

### 4. Checkpoints

Checkpoints are saved at regular intervals and the best model is stored based on validation accuracy.

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None
}
```

## How to Run

### 1. Train the Model

```bash
python main.py
```

- Modify `main.py` to adjust dataset paths, epochs, and batch sizes.
- Uncomment the `train_model` call to start training.

### 2. Evaluate the Model

Ensure the best model checkpoint is available:

```python
best_model_path = "/path/to/checkpoint/best_model.pth"
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint)
evaluate_model(model, test_loader, device)
```

### 3. Visualize Predictions

```python
visualize_predictions(model, transform, test_folder, save_path, num_images, device)
```

## Customization

- Modify `transform` for different augmentation strategies.
- Adjust the neural network architecture by changing the ResNet model or adding custom layers.

## Notes

- Ensure the test data transformations do not include random augmentations (use deterministic transforms)
- Always set the model to evaluation mode before testing:

```python
model.eval()
```

## Troubleshooting

- If evaluation results vary, ensure the model is in eval mode and no dropout or batch norm randomness affects predictions.
- Verify the correct checkpoint is loaded by printing checkpoint details.

## Acknowledgment

This project leverages PyTorch and pretrained models from Torchvision for UAV image classification.
