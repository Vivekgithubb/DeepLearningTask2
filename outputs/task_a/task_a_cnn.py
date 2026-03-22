"""
Task A: CNN Image Classification on Fashion-MNIST
Generates all report assets automatically into ./outputs/task_a/
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Output folder ─────────────────────────────────────────────────────────────
OUT = "./outputs/task_a"
os.makedirs(OUT, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 1e-3

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
# Custom CNN  → 28×28 grayscale
transform_custom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MobileNet   → 64×64, 3-channel
transform_mobile = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_loaders(transform):
    full_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set   = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(full_train))
    val_size   = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """Custom 2-block CNN with BatchNorm + Dropout"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 14×14
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 7×7
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 3×3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def get_mobilenet(num_classes=10):
    """Pretrained MobileNetV2, frozen base, new head"""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, val_loader, name, lr=LR):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epoch_times = []

    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAvg time/epoch: {avg_epoch_time:.1f}s")
    return model, history, avg_epoch_time

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTTING HELPERS  (all saved to disk)
# ─────────────────────────────────────────────────────────────────────────────
def plot_curves(history_a, history_b, label_a, label_b):
    """Training & validation curves for both models side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, EPOCHS + 1)

    # Loss
    axes[0].plot(epochs, history_a['train_loss'], label=f'{label_a} Train')
    axes[0].plot(epochs, history_a['val_loss'],   label=f'{label_a} Val', linestyle='--')
    axes[0].plot(epochs, history_b['train_loss'], label=f'{label_b} Train')
    axes[0].plot(epochs, history_b['val_loss'],   label=f'{label_b} Val', linestyle='--')
    axes[0].set_title('Loss Curves'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history_a['train_acc'], label=f'{label_a} Train')
    axes[1].plot(epochs, history_a['val_acc'],   label=f'{label_a} Val', linestyle='--')
    axes[1].plot(epochs, history_b['train_acc'], label=f'{label_b} Train')
    axes[1].plot(epochs, history_b['val_acc'],   label=f'{label_b} Val', linestyle='--')
    axes[1].set_title('Accuracy Curves'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = f"{OUT}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_confusion_matrix(preds, labels, model_name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f'Confusion Matrix – {model_name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_')
    path  = f"{OUT}/confusion_matrix_{fname}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── MODEL 1: Custom CNN ───────────────────────────────────────────────────
    train_c, val_c, test_c = get_loaders(transform_custom)
    cnn_model = SimpleCNN()
    cnn_model, cnn_history, cnn_epoch_time = train_model(
        cnn_model, train_c, val_c, "Simple CNN")
    _, cnn_test_acc, cnn_preds, cnn_labels = evaluate(
        cnn_model, test_c, nn.CrossEntropyLoss())
    print(f"\nSimple CNN Test Accuracy: {cnn_test_acc:.4f}")

    # ── MODEL 2: MobileNet ────────────────────────────────────────────────────
    train_m, val_m, test_m = get_loaders(transform_mobile)
    mob_model = get_mobilenet()
    mob_model, mob_history, mob_epoch_time = train_model(
        mob_model, train_m, val_m, "MobileNet", lr=1e-4)
    _, mob_test_acc, mob_preds, mob_labels = evaluate(
        mob_model, test_m, nn.CrossEntropyLoss())
    print(f"\nMobileNet Test Accuracy: {mob_test_acc:.4f}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    plot_curves(cnn_history, mob_history, "Simple CNN", "MobileNet")
    plot_confusion_matrix(cnn_preds, cnn_labels, "Simple CNN")
    plot_confusion_matrix(mob_preds, mob_labels, "MobileNet")

    # ── CLASSIFICATION REPORTS ────────────────────────────────────────────────
    for name, preds, labels in [("Simple_CNN", cnn_preds, cnn_labels),
                                 ("MobileNet",  mob_preds, mob_labels)]:
        report = classification_report(labels, preds, target_names=CLASS_NAMES)
        path = f"{OUT}/classification_report_{name}.txt"
        with open(path, 'w') as f:
            f.write(report)
        print(f"\n{name} Classification Report:\n{report}")
        print(f"Saved: {path}")

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    cnn_params = count_parameters(cnn_model)
    mob_params = count_parameters(mob_model)
    mob_trainable = count_trainable(mob_model)

    comparison = {
        "Simple CNN": {
            "test_accuracy":      round(cnn_test_acc, 4),
            "total_parameters":   cnn_params,
            "trainable_params":   cnn_params,
            "avg_epoch_time_sec": round(cnn_epoch_time, 1)
        },
        "MobileNet": {
            "test_accuracy":      round(mob_test_acc, 4),
            "total_parameters":   mob_params,
            "trainable_params":   mob_trainable,
            "avg_epoch_time_sec": round(mob_epoch_time, 1)
        }
    }

    cmp_path = f"{OUT}/model_comparison.json"
    with open(cmp_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for model_name, stats in comparison.items():
        print(f"\n{model_name}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print(f"\nSaved comparison: {cmp_path}")

    # ── SAVE MODELS ───────────────────────────────────────────────────────────
    torch.save(cnn_model.state_dict(), f"{OUT}/simple_cnn.pth")
    torch.save(mob_model.state_dict(), f"{OUT}/mobilenet.pth")
    print(f"\nAll outputs saved to: {OUT}/")
    print("\nFiles saved:")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")
