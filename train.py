"""
Training pipeline for binary classification (YES vs NO) on Google Speech Commands
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from melbanks import LogMelFilterBanks


class BinarySpeechCommands(SPEECHCOMMANDS):
    """Dataset wrapper for binary classification (YES vs NO)"""
    
    def __init__(self, root: str, subset: str = 'training'):
        super().__init__(root, download=True, subset=subset)
        self.classes = ['yes', 'no']
        self.class_to_idx = {'yes': 0, 'no': 1}
        
        self.filtered_indices = []
        for idx in range(len(self._walker)):
            filepath = self._walker[idx]
            label = os.path.basename(os.path.dirname(filepath))
            if label in self.classes:
                self.filtered_indices.append(idx)
        
        print(f"Loaded {len(self.filtered_indices)} samples for {subset} subset")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        waveform, sample_rate, label, _, _ = super().__getitem__(actual_idx)
        label_idx = self.class_to_idx[label]
        return waveform.squeeze(0), label_idx


class PadCollate:
    """Collate function to pad waveforms to the same length"""
    
    def __init__(self, max_length: int = 16000):
        self.max_length = max_length
    
    def __call__(self, batch):
        waveforms, labels = zip(*batch)
        padded = []
        for w in waveforms:
            if w.shape[0] < self.max_length:
                w = torch.nn.functional.pad(w, (0, self.max_length - w.shape[0]))
            else:
                w = w[:self.max_length]
            padded.append(w)
        return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


class SimpleCNN(nn.Module):
    """Simple CNN model for audio classification using Conv1d"""
    
    def __init__(self, n_mels: int = 80, groups: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.logmel = LogMelFilterBanks(n_mels=n_mels)
        
        # Calculate actual groups for each layer
        g1 = min(groups, n_mels) if groups > 1 and n_mels % min(groups, n_mels) == 0 else 1
        g2 = min(groups, 32) if groups > 1 and 32 % min(groups, 32) == 0 else 1
        g3 = min(groups, 64) if groups > 1 and 64 % min(groups, 64) == 0 else 1
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_mels, 32, 3, padding=1, groups=g1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1, groups=g2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1, groups=g3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.logmel(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, input_length: int = 16000) -> int:
        """
        Estimate FLOPs for the model
        For Conv1d: FLOPs = 2 * output_length * in_channels * out_channels * kernel_size / groups
        For Linear: FLOPs = 2 * input_features * output_features
        """
        total_flops = 0
        
        # LogMelFilterBanks: STFT + log + mel
        # STFT: ~5 * n_fft * input_length FLOPs (approximation)
        n_fft = 512
        total_flops += 5 * n_fft * input_length  # STFT
        total_flops += input_length // 2 * self.logmel.n_mels  # Mel filterbank
        total_flops += input_length // 2 * self.logmel.n_mels  # Log operation
        
        # Conv1d layers
        # After logmel: (input_length // 2) frames
        length = input_length // 2
        
        # Conv1: n_mels -> 32, kernel=3, padding=1
        g1 = min(self.conv1[0].groups, self.logmel.n_mels) if self.conv1[0].groups > 1 else 1
        out_length1 = length  # after padding
        flops1 = 2 * out_length1 * self.logmel.n_mels * 32 * 3 // g1
        total_flops += flops1
        # After MaxPool1d(2): length //= 2
        length = length // 2
        
        # Conv2: 32 -> 64, kernel=3, padding=1
        g2 = min(self.conv2[0].groups, 32) if self.conv2[0].groups > 1 else 1
        out_length2 = length
        flops2 = 2 * out_length2 * 32 * 64 * 3 // g2
        total_flops += flops2
        # After MaxPool1d(2): length //= 2
        length = length // 2
        
        # Conv3: 64 -> 128, kernel=3, padding=1
        g3 = min(self.conv3[0].groups, 64) if self.conv3[0].groups > 1 else 1
        out_length3 = length
        flops3 = 2 * out_length3 * 64 * 128 * 3 // g3
        total_flops += flops3
        # After MaxPool1d(2): length //= 2
        length = length // 2
        
        # Global Average Pooling: just averaging, minimal FLOPs
        total_flops += 128
        
        # FC: 128 -> 2
        flops_fc = 2 * 128 * 2
        total_flops += flops_fc
        
        return total_flops


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_experiment(n_mels, groups, data_dir, output_dir, epochs=5, batch_size=64):
    print(f"\n{'='*50}")
    print(f"Running: n_mels={n_mels}, groups={groups}")
    print(f"{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = BinarySpeechCommands(root=data_dir, subset='training')
    val_ds = BinarySpeechCommands(root=data_dir, subset='validation')
    test_ds = BinarySpeechCommands(root=data_dir, subset='testing')
    
    collate = PadCollate()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate)
    
    model = SimpleCNN(n_mels=n_mels, groups=groups).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    params = model.count_parameters()
    flops = model.count_flops()
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops:,}")
    
    train_losses = []
    val_accs = []
    epoch_times = []
    
    for epoch in range(epochs):
        start = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - start
        
        train_losses.append(loss)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, ValAcc={val_acc:.4f}, Time={epoch_time:.1f}s")
    
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return {
        'n_mels': n_mels,
        'groups': groups,
        'parameters': params,
        'flops': flops,
        'test_accuracy': test_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accs,
        'epoch_times': epoch_times
    }


def main():
    print("="*50)
    print("Speech Commands Binary Classification Experiments")
    print("="*50)
    
    data_dir = './speech_commands'
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # Experiments: n_mels variations
    for n_mels in [20, 40, 80]:
        result = run_experiment(n_mels, 1, data_dir, output_dir, epochs=5)
        all_results.append(result)
    
    # Experiments: groups variations
    for groups in [2, 4, 8]:
        result = run_experiment(80, groups, data_dir, output_dir, epochs=5)
        all_results.append(result)
    
    # Save results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in all_results:
        print(f"n_mels={r['n_mels']:2d}, groups={r['groups']:2d}: TestAcc={r['test_accuracy']:.4f}, Params={r['parameters']:,}, FLOPs={r['flops']:,}")


if __name__ == '__main__':
    main()