# Vision Transformer from Scratch

This project is an implementation of the Vision Transformer (ViT) model from scratch using PyTorch. The model is trained to perform image classification on the CIFAR-10 dataset.

## Features

- Implementation of ViT components from scratch:
  - Patch Embedding
  - Transformer Encoder with Multi-Head Self-Attention
  - MLP Head
- Training and evaluation scripts for CIFAR-10.
- Data augmentation for better generalization.
- Saves the best model based on validation loss.
- Visualizes training progress (accuracy and loss curves).
- Visualizes model predictions on test images.

## Model Architecture

The Vision Transformer model consists of the following components:

1.  **Patch Embedding:** The input image is divided into fixed-size patches, which are then linearly embedded. A learnable `[CLS]` token is prepended to the sequence of patch embeddings. Positional embeddings are added to retain positional information.
2.  **Transformer Encoder:** The sequence of embeddings is processed by a series of Transformer encoder blocks. Each block consists of:
    - Layer Normalization
    - Multi-Head Self-Attention
    - Layer Normalization
    - MLP (Multi-Layer Perceptron)
3.  **Classification Head:** The output corresponding to the `[CLS]` token is passed through a final MLP head to produce the classification scores.

## Dataset

The model is trained on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60,000 32x32 color images in 10 classes.

## Requirements

The code is written in Python and requires the following libraries:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm`

You can install the dependencies using pip. It is recommended to install PyTorch from the [official website](https://pytorch.org/get-started/locally/) to ensure compatibility with your system's CUDA version if you have a GPU.

```bash
pip install numpy matplotlib tqdm
```

## How to Run

1.  Clone the repository.
2.  Install the required libraries (see Requirements).
3.  Run the Python script:
    ```bash
    python vision_transformer_from_scratch.py
    ```

The script will automatically download the CIFAR-10 dataset, train the model, save the best model weights to `vision_transformer_best_model.pth`, and display the training plots and sample predictions.

## Hyperparameters

The key hyperparameters are defined in the script and can be modified:

- `BATCH_SIZE`: 128
- `EPOCHS`: 30
- `LEARNING_RATE`: 3e-4
- `PATCH_SIZE`: 4
- `IMAGE_SIZE`: 32
- `EMBED_DIM`: 256
- `NUM_HEADS`: 8
- `DEPTH`: 6 (Number of Transformer Encoder layers)
- `MLP_DIM`: 512

## Results

After training, the script will:

1.  Plot the training and validation accuracy and loss curves.
2.  Display a grid of random test images with their predicted and true labels.
