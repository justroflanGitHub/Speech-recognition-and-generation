# Digital Signal Processing Assignment Report

## LogMelFilterBanks Implementation and CNN Classification on Google Speech Commands

## 1. Introduction

This assignment implements a PyTorch layer for extracting logarithms of Mel-scale filterbank energies (LogMelFilterBanks) using basic torch operations. The implementation is then used as a feature extraction layer in a CNN for binary classification of "YES" vs "NO" commands from the Google Speech Commands dataset.

---

## 2. LogMelFilterBanks Implementation

### 2.1 Architecture

The `LogMelFilterBanks` class inherits from `torch.nn.Module` and implements the following pipeline:

1. **STFT**: Short-Time Fourier Transform using Hann window
2. **Power Spectrum**: Magnitude squared of complex STFT output
3. **Mel Filterbank**: Triangular filterbank matrix from `torchaudio.functional.melscale_fbanks`
4. **Logarithm**: Natural log with epsilon (1e-6) for numerical stability

### 2.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_fft | 400 | FFT window size |
| hop_length | 160 | Hop length between frames |
| n_mels | 80 | Number of Mel filterbanks |
| samplerate | 16000 | Audio sample rate |
| power | 2.0 | Power exponent for spectrum |

### 2.3 Verification

The implementation was verified against `torchaudio.transforms.MelSpectrogram`:

```python
melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
logmelbanks = LogMelFilterBanks()(signal)

assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
```

---

## 3. CNN Model Architecture

### 3.1 Model Design

```
Input (batch, 16000)
    ↓
LogMelFilterBanks (batch, n_mels, 101)
    ↓
Conv1d(n_mels → 32) + BN + ReLU + MaxPool + Dropout
    ↓
Conv1d(32 → 64) + BN + ReLU + MaxPool + Dropout
    ↓
Conv1d(64 → 128) + BN + ReLU + MaxPool + Dropout
    ↓
AdaptiveAvgPool1d
    ↓
Linear(128 → 2)
```

### 3.2 Parameter Count

The model is designed to stay under 100K parameters:
- Default (n_mels=80, groups=1): **39,330 parameters**
- Minimum (n_mels=80, groups=8): **5,730 parameters**

---

## 4. Experiments

### 4.1 Dataset

- **Dataset**: Google Speech Commands
- **Classes**: "YES" vs "NO" (binary classification)
- **Training samples**: 6,358
- **Validation samples**: 803
- **Test samples**: 824
- **Sample rate**: 16,000 Hz
- **Duration**: 1 second (padded/truncated)

### 4.2 Training Configuration

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss
- **Batch size**: 64
- **Epochs**: 5

---

## 5. Results

### 5.1 Experiment 1: Varying n_mels

| n_mels | groups | Parameters | FLOPs | Test Accuracy | Avg Epoch Time |
|--------|--------|------------|-------|---------------|----------------|
| 20 | 1 | 33,570 | 219.5M | 98.42% | 2.0s |
| 40 | 1 | 35,490 | 250.5M | 98.54% | 1.9s |
| 80 | 1 | 39,330 | 312.6M | 97.94% | 2.0s |

![n_mels experiment](results/n_mels_experiment.png)

**Observations**:
- All configurations achieve similar accuracy (~98%)
- n_mels=20 is sufficient for this binary task with lowest FLOPs (219.5M)
- Training time is consistent across n_mels values
- Best accuracy: n_mels=40 with 98.54%

### 5.2 Experiment 2: Varying groups

| n_mels | groups | Parameters | FLOPs | Test Accuracy | Avg Epoch Time |
|--------|--------|------------|-------|---------------|----------------|
| 80 | 1 | 39,330 | 312.6M | 97.94% | 2.0s |
| 80 | 2 | 20,130 | 177.4M | 96.24% | 2.1s |
| 80 | 4 | 10,530 | 109.8M | 93.81% | 2.1s |
| 80 | 8 | 5,730 | 76.0M | 91.14% | 2.0s |

![groups experiment](results/groups_experiment.png)

**Observations**:
- Grouped convolutions significantly reduce both parameters (up to 85% reduction) and FLOPs (up to 76% reduction)
- Moderate accuracy decrease with higher groups
- Training time not significantly affected
- groups=2 offers good trade-off: 43% fewer parameters, 43% fewer FLOPs, only 1.7% accuracy drop

---

## 6. Conclusions

### 6.1 LogMelFilterBanks

The implementation successfully replicates the behavior of `torchaudio.transforms.MelSpectrogram` with log scaling. Using only basic PyTorch operations makes it portable and easy to integrate into any PyTorch pipeline.

### 6.2 Model Performance

- The simple CNN architecture achieves ~97-98% accuracy on binary classification
- All n_mels configurations perform similarly, suggesting 20 Mel bands is sufficient for this task
- Grouped convolutions offer a good trade-off between model size, FLOPs, and accuracy
- FLOPs reduction closely tracks parameter reduction for grouped convolutions

### 6.3 Recommendations

| Configuration | Parameters | FLOPs | Test Accuracy | Use Case |
|--------------|------------|-------|---------------|----------|
| n_mels=40, groups=1 | 35,490 | 250.5M | 98.54% | Best accuracy |
| n_mels=20, groups=1 | 33,570 | 219.5M | 98.42% | Best efficiency |
| n_mels=80, groups=2 | 20,130 | 177.4M | 96.24% | Balanced |
| n_mels=80, groups=8 | 5,730 | 76.0M | 91.14% | Minimal size |

---

## 7. Figures

- **Figure 1**: Mel spectrogram comparison (custom vs torchaudio)
- **Figure 2**: Training loss and validation accuracy for n_mels experiments
- **Figure 3**: Parameters, accuracy, and training time for groups experiments

![summary table](results/summary_table.png)

---

