"""
Task C: GAN on Fashion-MNIST
Saves all report assets to ./outputs/task_c/
  - generated samples grid every N epochs  → sample_epoch_XXX.png
  - final progression montage              → progression.png
  - training loss curve                    → loss_curve.png
  - loss values                            → training_losses.json
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

OUT = "./outputs/task_c"
SAMPLES_DIR = f"{OUT}/samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE   = 28 * 28          # 784
BATCH_SIZE = 128
EPOCHS     = 100
LR_D       = 2e-4             # discriminator lr
LR_G       = 2e-4             # generator lr
SAVE_EVERY = 10               # save sample grid every N epochs
FIXED_ROWS = 5
FIXED_COLS = 5                # 5×5 = 25 fixed noise vectors for progression

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA  (normalize to [-1, 1] for Tanh output)
# ─────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])          # → [-1, 1]
])

dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)

# Fixed noise — same 25 vectors every save, so progression is comparable
fixed_noise = torch.randn(FIXED_ROWS * FIXED_COLS, LATENT_DIM, device=DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """Noise (100) → 784 → reshape to 28×28"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, momentum=0.8),
            # Block 2
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, momentum=0.8),
            # Block 3
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),
            # Output
            nn.Linear(1024, IMG_SIZE),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    """28×28 image → real / fake probability"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# Print param counts
g_params = sum(p.numel() for p in G.parameters())
d_params = sum(p.numel() for p in D.parameters())
print(f"Generator params:      {g_params:,}")
print(f"Discriminator params:  {d_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS + OPTIMIZERS
#    Label smoothing on real labels (0.9 instead of 1.0) → stabilizes D
# ─────────────────────────────────────────────────────────────────────────────
criterion  = nn.BCELoss()
opt_D      = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
opt_G      = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))

REAL_LABEL = 0.9    # label smoothing – mitigation for training instability
FAKE_LABEL = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE SAMPLE GRID
# ─────────────────────────────────────────────────────────────────────────────
def save_sample_grid(epoch, noise, tag=""):
    G.eval()
    with torch.no_grad():
        imgs = G(noise).cpu().numpy()          # (25, 1, 28, 28)
    imgs = (imgs * 0.5 + 0.5)                  # → [0, 1]

    fig = plt.figure(figsize=(FIXED_COLS * 1.2, FIXED_ROWS * 1.2))
    gs  = gridspec.GridSpec(FIXED_ROWS, FIXED_COLS, wspace=0.05, hspace=0.05)
    for i in range(FIXED_ROWS * FIXED_COLS):
        ax = fig.add_subplot(gs[i])
        ax.imshow(imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    label = f"Epoch {epoch}" + (f" – {tag}" if tag else "")
    fig.suptitle(label, fontsize=10, y=1.01)
    path = f"{SAMPLES_DIR}/sample_epoch_{epoch:03d}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    G.train()
    return path

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP  (alternating D → G)
# ─────────────────────────────────────────────────────────────────────────────
g_losses, d_losses = [], []
epoch_g, epoch_d   = [], []     # per-epoch averages

print(f"\n{'='*55}\nStarting GAN Training  ({EPOCHS} epochs)\n{'='*55}")

for epoch in range(1, EPOCHS + 1):
    batch_g, batch_d = [], []
    t0 = time.time()

    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(DEVICE)
        bsz       = real_imgs.size(0)

        real_labels = torch.full((bsz, 1), REAL_LABEL, device=DEVICE)
        fake_labels = torch.full((bsz, 1), FAKE_LABEL, device=DEVICE)

        # ── Step 1: Train Discriminator ──────────────────────────────────────
        D.zero_grad()

        # on real
        out_real = D(real_imgs)
        loss_real = criterion(out_real, real_labels)

        # on fake
        z         = torch.randn(bsz, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z).detach()           # detach so G gets no grad here
        out_fake  = D(fake_imgs)
        loss_fake = criterion(out_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # ── Step 2: Train Generator ──────────────────────────────────────────
        G.zero_grad()

        z         = torch.randn(bsz, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)
        out_fake  = D(fake_imgs)
        # G wants D to think fakes are real
        loss_G    = criterion(out_fake, real_labels)
        loss_G.backward()
        opt_G.step()

        batch_d.append(loss_D.item())
        batch_g.append(loss_G.item())

    # ── Epoch bookkeeping ─────────────────────────────────────────────────────
    avg_d = np.mean(batch_d)
    avg_g = np.mean(batch_g)
    epoch_d.append(avg_d)
    epoch_g.append(avg_g)
    elapsed = time.time() - t0

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | "
              f"Time: {elapsed:.1f}s")

    # ── Save sample grid ──────────────────────────────────────────────────────
    if epoch % SAVE_EVERY == 0 or epoch == 1:
        path = save_sample_grid(epoch, fixed_noise)
        print(f"  → Saved sample grid: {path}")

# Save final epoch regardless
save_sample_grid(EPOCHS, fixed_noise, tag="Final")

# ─────────────────────────────────────────────────────────────────────────────
# 6. LOSS CURVE
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curve():
    epochs = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, epoch_d, label='Discriminator Loss', color='#e74c3c', linewidth=1.5)
    ax.plot(epochs, epoch_g, label='Generator Loss',     color='#3498db', linewidth=1.5)
    ax.axhline(y=0.693, color='gray', linestyle=':', linewidth=1,
               label='Ideal equilibrium (ln2 ≈ 0.693)')
    ax.set_title('GAN Training Losses')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUT}/loss_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

plot_loss_curve()

# ─────────────────────────────────────────────────────────────────────────────
# 7. PROGRESSION MONTAGE  (epoch 1 / 10 / 30 / 60 / 100 side by side)
# ─────────────────────────────────────────────────────────────────────────────
def build_progression_montage():
    checkpoints = [1, 10, 30, 60, 100]
    # only include epochs we actually saved
    checkpoints = [e for e in checkpoints
                   if os.path.exists(f"{SAMPLES_DIR}/sample_epoch_{e:03d}.png")]

    fig, axes = plt.subplots(1, len(checkpoints),
                             figsize=(len(checkpoints) * 3, 3.5))
    if len(checkpoints) == 1:
        axes = [axes]

    for ax, ep in zip(axes, checkpoints):
        img_path = f"{SAMPLES_DIR}/sample_epoch_{ep:03d}.png"
        img      = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(f"Epoch {ep}", fontsize=11)
        ax.axis('off')

    plt.suptitle('Sample Quality Progression', fontsize=13, y=1.02)
    plt.tight_layout()
    path = f"{OUT}/progression.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

build_progression_montage()

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE LOSSES + MODELS
# ─────────────────────────────────────────────────────────────────────────────
losses_data = {
    "generator_losses":     [round(v, 6) for v in epoch_g],
    "discriminator_losses": [round(v, 6) for v in epoch_d],
    "final_g_loss":  round(epoch_g[-1], 4),
    "final_d_loss":  round(epoch_d[-1], 4),
    "generator_params":     g_params,
    "discriminator_params": d_params,
    "hyperparameters": {
        "latent_dim":   LATENT_DIM,
        "batch_size":   BATCH_SIZE,
        "epochs":       EPOCHS,
        "lr_G":         LR_G,
        "lr_D":         LR_D,
        "label_smoothing": REAL_LABEL,
        "optimizer":    "Adam(beta1=0.5, beta2=0.999)"
    }
}
with open(f"{OUT}/training_losses.json", 'w') as f:
    json.dump(losses_data, f, indent=2)

torch.save(G.state_dict(), f"{OUT}/generator.pth")
torch.save(D.state_dict(), f"{OUT}/discriminator.pth")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("TRAINING COMPLETE")
print("="*55)
print(f"Final Generator Loss:     {epoch_g[-1]:.4f}")
print(f"Final Discriminator Loss: {epoch_d[-1]:.4f}")
print(f"\nAll outputs saved to: {OUT}/")
print("\nFiles saved:")
for fname in sorted(os.listdir(OUT)):
    size = os.path.getsize(f"{OUT}/{fname}")
    print(f"  {fname:<40} ({size/1024:.1f} KB)")
print(f"\nSample grids ({len(os.listdir(SAMPLES_DIR))}) saved in: {SAMPLES_DIR}/")
