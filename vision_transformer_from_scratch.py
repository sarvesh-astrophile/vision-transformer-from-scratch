# %% [markdown]
# Installing required libraries
# !pip install matplotlib tqdm

# %% [markdown]
### 1. Importing required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import torchvision
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% [markdown]
# Check version of torch and torchvision
print(f"Torch version: {torch.__version__}")
# print(f"Torchvision version: {torchvision.__version__}")

# %% [markdown]
### 2. Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
### 3. Set the seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

# %% [markdown]
### 4. Setting up the hyperparameters
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1

# %% [markdown]
### 5. Define the data transformation
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# %% [markdown]
### 6. Load the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# %% [markdown]
### 7. Load the dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
### 8. Check the shape of the dataset
print(f"Total Train dataset batches: {len(train_loader)} and each batch size: {BATCH_SIZE}")
print(f"Total Test dataset batches: {len(test_loader)} and each batch size: {BATCH_SIZE}")

# %% [markdown]
### 9. Building the Vision Transformer model
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
            )
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) # (B, embed_dim, H, W)
        x = x.flatten(2) # (B, embed_dim, H*W)
        x = x.transpose(1, 2) # (B, H*W, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1) # (B, 1 + H*W, embed_dim)
        x = x + self.pos_embed # (B, 1 + H*W, embed_dim)
        return x

# %% [markdown]
### 10. MLP Block
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

# %% [markdown]
### 11. Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
# %% [markdown]
### 12. Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, mlp_dim, depth, drop_rate=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_dim, drop_rate) 
            for _ in range(depth)
            ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.head(cls_token)
        return x
 
# %% [markdown]
### 13. Initialize the model
model = VisionTransformer(
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=CHANNELS,
    num_classes=NUM_CLASSES,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    depth=DEPTH,
    drop_rate=DROP_RATE
    ).to(device)

# %% [markdown]
### 14. Print the model summary
print(model)
print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

# %% [markdown]
### 15. Define the optimizer and the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %% [markdown]
### 16. Define the training loop
def train_model(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for (data, target) in train_loader:
        # 1. Move the data to the device
        data, target = data.to(device), target.to(device) # (B, 3, 32, 32)
        # 2. Zero the gradients
        optimizer.zero_grad()
        # 3. Forward pass
        output = model(data)
        # 4. Calculate the loss
        loss = criterion(output, target)
        # 5. Backward pass
        loss.backward()
        # 6. Update the weights
        optimizer.step()
        # 7. Update the loss
        total_loss += loss.item() * data.size(0) # (B, 3, 32, 32)
        # 8. Update the accuracy
        correct += (output.argmax(dim=1) == target).sum().item()
    # 11. Return the loss and the accuracy
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

# %% [markdown]
### 17. Define the validation loop
def validate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.inference_mode():
        for (data, target) in test_loader:
            # 1. Move the data to the device
            data, target = data.to(device), target.to(device) # (B, 3, 32, 32)
            # 2. Forward pass
            output = model(data)
            # 3. Calculate the loss
            loss = criterion(output, target)
            # 4. Update the loss
            total_loss += loss.item() * data.size(0) # (B, 3, 32, 32)
            # 5. Update the accuracy
            correct += (output.argmax(dim=1) == target).sum().item()
    # 8. Return the loss and the accuracy
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# %% [markdown]
### 18. Train the model
best_loss = float('inf')
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []
for epoch in tqdm(range(EPOCHS), desc="Training"):
    train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device, epoch)
    test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss {train_loss:.4f} | Train Accuracy {train_accuracy:.4f}% | Test Loss {test_loss:.4f} | Test Accuracy {test_accuracy:.4f}%")
    # save the checkpoint if the validation loss is the lowest
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "vision_transformer_best_model.pth")

# %% [markdown]
### 19. Load the best model
model.load_state_dict(torch.load("vision_transformer_best_model.pth"))

# %% [markdown]
### 20. Plot the training and validation accuracy with loss in two subplots in vertical format
plt.figure(figsize=(5, 10))
plt.subplot(2, 1, 1)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy")   
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss")
plt.legend()
plt.show()

# %% [markdown]
### 21. Predict and plot the grid of random images
def predict_and_plot_grid(model, dataset, classes, num_images=9, max_rows=3, fontsize=8):
    model.eval()
    with torch.inference_mode():
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            img, label = random.choice(dataset)
            output = model(img.unsqueeze(0).to(device))
            pred_label = classes[output.argmax(dim=1)]
            plt.subplot(max_rows, num_images // max_rows, i + 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(f"Pred: {pred_label} \n True: {classes[label]}", fontsize=fontsize, color="green" if pred_label == classes[label] else "red")
    plt.tight_layout()
    plt.show()
    
predict_and_plot_grid(model, test_dataset, test_dataset.classes, num_images=9, fontsize=15)

# %% [markdown]