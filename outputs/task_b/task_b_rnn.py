"""
Task B: RNN / LSTM / GRU – Airline Passenger Time-Series Prediction
Saves all report assets to ./outputs/task_b/
"""

import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

OUT    = "./outputs/task_b"
os.makedirs(OUT, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN     = 12        # use past 12 months to predict next month
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
BATCH_SIZE  = 16
EPOCHS      = 100
LR          = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA – download inline (no Kaggle needed)
# ─────────────────────────────────────────────────────────────────────────────
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

def load_data():
    try:
        df = pd.read_csv(URL, header=0, usecols=[1])
    except Exception:
        # Fallback: hardcoded classic dataset
        values = [
            112,118,132,129,121,135,148,148,136,119,104,118,
            115,126,141,135,125,149,170,170,158,133,114,140,
            145,150,178,163,172,178,199,199,184,162,146,166,
            171,180,193,181,183,218,230,242,209,191,172,194,
            196,196,236,235,229,243,264,272,237,211,180,201,
            204,188,235,227,234,264,302,293,259,229,203,229,
            242,233,267,269,270,315,364,347,312,274,237,278,
            284,277,317,313,318,374,413,405,355,306,271,306,
            315,301,356,348,355,422,465,467,404,347,305,336,
            340,318,362,348,363,435,491,505,404,359,310,337,
            360,342,406,396,420,472,548,559,463,407,362,405,
            417,391,419,461,472,535,622,606,508,461,390,432
        ]
        df = pd.DataFrame(values, columns=['Passengers'])
    return df.values.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────────────
class SequenceModel(nn.Module):
    """Unified model: swap cell_type to get RNN / LSTM / GRU"""
    def __init__(self, cell_type='LSTM', input_size=1,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.cell_type = cell_type
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell_type]
        self.rnn = rnn_cls(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)      # (batch, seq, hidden)
        out     = out[:, -1, :]   # last timestep
        return self.fc(out)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def train_model(cell_type, train_loader, val_loader):
    model     = SequenceModel(cell_type).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': []}
    epoch_times = []

    print(f"\n{'='*45}\nTraining: {cell_type}\n{'='*45}")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        tr_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # ── val ──
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred    = model(xb).squeeze()
                va_loss += criterion(pred, yb.squeeze()).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        epoch_times.append(time.time() - t0)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")

    avg_time = np.mean(epoch_times)
    return model, history, avg_time

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, actuals = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out = model(xb).squeeze().cpu().numpy()
        preds.extend(np.atleast_1d(out))
        actuals.extend(yb.squeeze().numpy())
    return np.array(preds), np.array(actuals)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    colors = {'RNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}
    for ax, (name, hist) in zip(axes, histories.items()):
        ep = range(1, len(hist['train_loss']) + 1)
        ax.plot(ep, hist['train_loss'], label='Train', color=colors[name])
        ax.plot(ep, hist['val_loss'],   label='Val',   color=colors[name], linestyle='--', alpha=0.7)
        ax.set_title(f'{name} Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('Training Stability – RNN vs LSTM vs GRU', fontsize=13, y=1.02)
    plt.tight_layout()
    path = f"{OUT}/loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_predictions(results, scaler, raw_data):
    """Overlay actual vs predicted for all three models"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = {'RNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}

    for ax, (name, res) in zip(axes, results.items()):
        # inverse transform
        actual = scaler.inverse_transform(res['actual'].reshape(-1, 1)).flatten()
        pred   = scaler.inverse_transform(res['pred'].reshape(-1, 1)).flatten()
        x      = range(len(actual))
        ax.plot(x, actual, label='Actual',    color='black',       linewidth=1.5)
        ax.plot(x, pred,   label='Predicted', color=colors[name],  linewidth=1.5, linestyle='--')
        ax.set_title(f'{name}  (RMSE={res["rmse"]:.2f})')
        ax.set_xlabel('Time Step'); ax.set_ylabel('Passengers')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Actual vs Predicted – Test Set', fontsize=13, y=1.02)
    plt.tight_layout()
    path = f"{OUT}/predictions.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_combined_prediction(results, scaler, raw_data, split_idx):
    """Single plot showing all three predictions vs actual"""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'RNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}

    # Full actual series
    full = scaler.inverse_transform(raw_data).flatten()
    ax.plot(range(len(full)), full, label='Actual', color='black', linewidth=2)

    # Each model's test predictions (placed at the right x position)
    test_start = split_idx  # approximate
    for name, res in results.items():
        pred = scaler.inverse_transform(res['pred'].reshape(-1, 1)).flatten()
        ax.plot(range(test_start, test_start + len(pred)),
                pred, label=f'{name} pred', color=colors[name],
                linestyle='--', linewidth=1.5)

    ax.axvline(x=split_idx, color='gray', linestyle=':', linewidth=1.5, label='Train/Test split')
    ax.set_title('Airline Passengers – All Models vs Actual')
    ax.set_xlabel('Month Index'); ax.set_ylabel('Passengers')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUT}/combined_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load & preprocess ─────────────────────────────────────────────────────
    raw      = load_data()
    scaled, scaler = preprocess(raw)
    X, y     = make_sequences(scaled, SEQ_LEN)

    # Train: first 120 samples, Test: remaining (~20 months)
    TRAIN_SIZE = 108
    X_train, y_train = X[:TRAIN_SIZE],  y[:TRAIN_SIZE]
    X_test,  y_test  = X[TRAIN_SIZE:],  y[TRAIN_SIZE:]

    # Further split train → 80/20 for validation
    val_split  = int(0.8 * len(X_train))
    X_tr, y_tr = X_train[:val_split], y_train[:val_split]
    X_va, y_va = X_train[val_split:], y_train[val_split:]

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    val_ds   = TimeSeriesDataset(X_va, y_va)
    test_ds  = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Device: {DEVICE}")

    # ── Train all three ───────────────────────────────────────────────────────
    histories  = {}
    results    = {}
    epoch_times = {}

    for cell in ['RNN', 'LSTM', 'GRU']:
        model, hist, avg_t = train_model(cell, train_loader, val_loader)
        preds, actuals     = predict(model, test_loader)
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        # RMSE in original scale
        pred_orig   = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
        actual_orig = scaler.inverse_transform(actuals.reshape(-1,1)).flatten()
        rmse_orig   = np.sqrt(mean_squared_error(actual_orig, pred_orig))

        histories[cell]   = hist
        epoch_times[cell] = round(avg_t, 3)
        results[cell]     = {
            'pred':      preds,
            'actual':    actuals,
            'rmse':      rmse_orig,
            'rmse_norm': round(rmse, 6),
        }
        print(f"\n{cell} Test RMSE (original scale): {rmse_orig:.2f} passengers")

        torch.save(model.state_dict(), f"{OUT}/{cell.lower()}_model.pth")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_loss_curves(histories)
    plot_predictions(results, scaler, raw)
    plot_combined_prediction(results, scaler, scaled, TRAIN_SIZE + SEQ_LEN)

    # ── Comparison JSON ───────────────────────────────────────────────────────
    comparison = {
        cell: {
            "rmse_original_scale": round(results[cell]['rmse'], 2),
            "rmse_normalized":     results[cell]['rmse_norm'],
            "avg_epoch_time_sec":  epoch_times[cell],
            "final_train_loss":    round(histories[cell]['train_loss'][-1], 6),
            "final_val_loss":      round(histories[cell]['val_loss'][-1], 6),
        }
        for cell in ['RNN', 'LSTM', 'GRU']
    }

    cmp_path = f"{OUT}/model_comparison.json"
    with open(cmp_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"{'Model':<8} {'RMSE':>10} {'Train Loss':>12} {'Val Loss':>10} {'Time/ep':>10}")
    print("-"*52)
    for cell, stats in comparison.items():
        print(f"{cell:<8} {stats['rmse_original_scale']:>10.2f} "
              f"{stats['final_train_loss']:>12.6f} "
              f"{stats['final_val_loss']:>10.6f} "
              f"{stats['avg_epoch_time_sec']:>9.3f}s")

    print(f"\nAll outputs saved to: {OUT}/")
    print("\nFiles saved:")
    for fname in sorted(os.listdir(OUT)):
        print(f"  {fname}")
