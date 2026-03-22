# Deep Learning Assignment — IS3332-1
### Applied Deep Models: CNN | RNN/LSTM/GRU | GAN

**Course:** Deep Learning (IS3332-1)  
**Department:** Information Science & Engineering  
**Dataset:** Fashion-MNIST · Airline Passengers  

---

## Overview

This repository contains the complete implementation for a three-part deep learning assignment covering the three core architectural families in modern deep learning:

| Task | Topic | Dataset |
|------|-------|---------|
| A | CNN Image Classification (Custom + Transfer Learning) | Fashion-MNIST |
| B | Time-Series Prediction (RNN / LSTM / GRU) | Airline Passengers |
| C | Image Generation (GAN) | Fashion-MNIST |

---

## Repository Structure

```
DeepLearningTask2/

│
└── outputs/
    ├── task_a/
    │   ├── training_curves.png
    │   ├── confusion_matrix_simple_cnn.png
    │   ├── confusion_matrix_mobilenet.png
    │   ├── classification_report_Simple_CNN.txt
    │   ├── classification_report_MobileNet.txt
    │   ├── model_comparison.json
    │   ├── simple_cnn.pth
    │   ├── task_a_cnn.py          # Task A — Custom CNN + MobileNetV2
    │   └── mobilenet.pth
    │
    ├── task_b/
    │   ├── loss_curves.png
    │   ├── predictions.png
    │   ├── combined_forecast.png
    │   ├── model_comparison.json
    │   ├── rnn_model.pth
    |   ├── task_b_rnn.py          # Task B — RNN, LSTM, GRU
    │   ├── lstm_model.pth
    │   └── gru_model.pth
    │
    └── task_c/
        ├── loss_curve.png
        ├── progression.png
        ├── training_losses.json
        ├── generator.pth
        ├── discriminator.pth
        ├── task_c_gan.py          # Task C — GAN training
        └── samples/
            ├── sample_epoch_001.png
            ├── sample_epoch_010.png
            ├── sample_epoch_030.png
            ├── sample_epoch_060.png
            └── sample_epoch_100.png
```

---

## Requirements

Python 3.10 or above is required.

Install all dependencies with:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy pandas
```

No GPU is required — all scripts run on CPU. If a CUDA-compatible GPU is available, it will be used automatically.

---

## Cloning the Repository

```bash
git clone https://github.com/Vivekgithubb/DeepLearningTask2.git
cd DeepLearningTask2
```

---

## Running the Code

### Task A — CNN Image Classification

```bash
python task_a_cnn.py
```

Trains a custom 3-block CNN and a pretrained MobileNetV2 on Fashion-MNIST for 15 epochs each. All output files are saved to `outputs/task_a/`.

**What gets saved:**
- `training_curves.png` — loss and accuracy curves for both models
- `confusion_matrix_*.png` — 10×10 confusion matrices
- `classification_report_*.txt` — per-class precision, recall, F1
- `model_comparison.json` — accuracy, parameters, training time
- `*.pth` — saved model weights

---

### Task B — RNN / LSTM / GRU Time-Series

```bash
python task_b_rnn.py
```

Downloads the Airline Passengers dataset automatically and trains RNN, LSTM, and GRU models for 100 epochs each. All output files are saved to `outputs/task_b/`.

**What gets saved:**
- `loss_curves.png` — MSE loss curves for all three models
- `predictions.png` — actual vs predicted on test set per model
- `combined_forecast.png` — all three predictions overlaid on full series
- `model_comparison.json` — RMSE, final losses, epoch times
- `*_model.pth` — saved model weights

---

### Task C — GAN Image Generation

```bash
python task_c_gan.py
```

Trains a fully connected GAN on Fashion-MNIST for 100 epochs. A 5×5 grid of generated images is saved every 10 epochs using fixed noise vectors. All output files are saved to `outputs/task_c/`.

**What gets saved:**
- `loss_curve.png` — Generator and Discriminator BCE losses
- `progression.png` — side-by-side quality comparison at epochs 1, 10, 30, 60, 100
- `samples/sample_epoch_XXX.png` — individual grids at each checkpoint
- `training_losses.json` — full loss history and hyperparameters
- `generator.pth` / `discriminator.pth` — saved model weights

---

## Task Summaries

### Task A — CNN Image Classification

Two CNN models are trained on Fashion-MNIST (70,000 grayscale clothing images, 10 classes):

- **Custom CNN:** Three Conv→BatchNorm→ReLU→MaxPool blocks followed by a Dropout+Linear classifier. Trained from scratch on 28×28 input. Achieved **92.33% test accuracy**.
- **MobileNetV2:** Pretrained on ImageNet, base layers frozen, only the 10-class head is fine-tuned. Achieved **81.86% test accuracy**.

The custom CNN outperforms transfer learning here due to domain mismatch — MobileNet's ImageNet features do not transfer well to low-resolution grayscale clothing images.

---

### Task B — Time-Series Prediction

Three recurrent models predict monthly airline passenger counts from a 12-month sliding window:

| Model | RMSE (passengers) |
|-------|-------------------|
| RNN   | 60.71 |
| LSTM  | 79.18 |
| GRU   | **43.13** |

GRU achieves the best RMSE. LSTM underperforms on this small dataset (144 samples) due to its higher parameter count causing overfitting. Vanilla RNN performs competitively on the 12-step window where vanishing gradients are less severe.

---

### Task C — GAN Image Generation

A fully connected GAN is trained on Fashion-MNIST using alternating Discriminator→Generator updates:

- **Generator:** 100-dim noise → Linear blocks with BatchNorm → 28×28 image (Tanh output)
- **Discriminator:** 28×28 image → Linear blocks with Dropout → real/fake probability (Sigmoid)

Both losses converge stably by epoch 30. Generated samples at epoch 100 show diverse, recognisable clothing items across 6+ categories with no mode collapse.

**Failure mode observed:** Discriminator dominance in early epochs caused Generator loss to spike, starving it of useful gradients.  
**Mitigations applied:** Label smoothing (real=0.9), Adam β₁=0.5, Dropout in Discriminator.

---

## Results Summary

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| A | Custom CNN | Test Accuracy | 92.33% |
| A | MobileNetV2 | Test Accuracy | 81.86% |
| B | RNN | RMSE | 60.71 |
| B | LSTM | RMSE | 79.18 |
| B | GRU | RMSE | **43.13** |
| C | GAN | Final G Loss | 0.7887 |
| C | GAN | Final D Loss | 1.3734 |

---

## Hyperparameters

| Parameter | Task A (CNN) | Task A (Mobile) | Task B | Task C |
|-----------|-------------|----------------|--------|--------|
| Batch Size | 64 | 64 | 16 | 128 |
| Epochs | 15 | 15 | 100 | 100 |
| Learning Rate | 0.001 | 0.0001 | 0.001 | 0.0002 |
| Optimizer | Adam | Adam | Adam | Adam |
| Loss | CrossEntropy | CrossEntropy | MSE | BCE |

---

## References

1. I. Goodfellow et al., *Deep Learning*, MIT Press, 2016.
2. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
3. I. Goodfellow et al., "Generative Adversarial Nets," *NeurIPS*, 2014.
4. H. Xiao et al., "Fashion-MNIST," arXiv:1708.07747, 2017.
5. PyTorch Documentation — https://pytorch.org/docs
