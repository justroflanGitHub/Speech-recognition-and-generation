"""
Generate plots for experiment results
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_mel_comparison(results, output_dir):
    """Plot comparison for varying n_mels"""
    n_mels_results = [r for r in results if r['groups'] == 1]
    n_mels_results.sort(key=lambda x: x['n_mels'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss curves
    ax1 = axes[0, 0]
    for r in n_mels_results:
        ax1.plot(range(1, len(r['train_losses'])+1), r['train_losses'], 
                marker='o', label=f"n_mels={r['n_mels']}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs Epoch (varying n_mels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy curves
    ax2 = axes[0, 1]
    for r in n_mels_results:
        ax2.plot(range(1, len(r['val_accuracies'])+1), r['val_accuracies'],
                marker='o', label=f"n_mels={r['n_mels']}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy vs Epoch (varying n_mels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: n_mels vs test accuracy
    ax3 = axes[1, 0]
    n_mels_vals = [r['n_mels'] for r in n_mels_results]
    test_accs = [r['test_accuracy'] for r in n_mels_results]
    bars = ax3.bar([str(nm) for nm in n_mels_vals], test_accs, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax3.set_xlabel('Number of Mel Filterbanks')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Test Accuracy vs n_mels')
    ax3.set_ylim(0.9, 1.0)
    for bar, acc in zip(bars, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{acc:.4f}', ha='center', fontsize=10)
    
    # Plot 4: n_mels vs parameters
    ax4 = axes[1, 1]
    params = [r['parameters'] for r in n_mels_results]
    bars = ax4.bar([str(nm) for nm in n_mels_vals], params, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax4.set_xlabel('Number of Mel Filterbanks')
    ax4.set_ylabel('Parameters')
    ax4.set_title('Model Parameters vs n_mels')
    for bar, p in zip(bars, params):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02, 
                f'{p:,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'n_mels_experiment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved n_mels experiment plot")

def plot_groups_comparison(results, output_dir):
    """Plot comparison for varying groups"""
    groups_results = [r for r in results if r['n_mels'] == 80]
    groups_results.sort(key=lambda x: x['groups'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Epoch training time vs groups
    ax1 = axes[0, 0]
    groups_vals = [r['groups'] for r in groups_results]
    avg_epoch_times = [np.mean(r['epoch_times']) for r in groups_results]
    bars = ax1.bar([str(g) for g in groups_vals], avg_epoch_times, color='#27ae60')
    ax1.set_xlabel('Groups Parameter')
    ax1.set_ylabel('Average Epoch Time (seconds)')
    ax1.set_title('Epoch Training Time vs Groups')
    for bar, t in zip(bars, avg_epoch_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_epoch_times)*0.02, 
                f'{t:.1f}s', ha='center', fontsize=10)
    
    # Plot 2: Parameters vs groups
    ax2 = axes[0, 1]
    params = [r['parameters'] for r in groups_results]
    bars = ax2.bar([str(g) for g in groups_vals], params, color='#2980b9')
    ax2.set_xlabel('Groups Parameter')
    ax2.set_ylabel('Parameters')
    ax2.set_title('Model Parameters vs Groups')
    for bar, p in zip(bars, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02, 
                f'{p:,}', ha='center', fontsize=10)
    
    # Plot 3: Test accuracy vs groups
    ax3 = axes[1, 0]
    test_accs = [r['test_accuracy'] for r in groups_results]
    bars = ax3.bar([str(g) for g in groups_vals], test_accs, color='#8e44ad')
    ax3.set_xlabel('Groups Parameter')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Test Accuracy vs Groups')
    ax3.set_ylim(0.9, 1.0)
    for bar, acc in zip(bars, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{acc:.4f}', ha='center', fontsize=10)
    
    # Plot 4: Training loss curves for different groups
    ax4 = axes[1, 1]
    for r in groups_results:
        ax4.plot(range(1, len(r['train_losses'])+1), r['train_losses'],
                marker='o', label=f"groups={r['groups']}")
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Loss')
    ax4.set_title('Training Loss vs Epoch (varying groups)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'groups_experiment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved groups experiment plot")

def plot_summary_table(results, output_dir):
    """Create a summary table"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    headers = ['n_mels', 'groups', 'Parameters', 'Test Acc', 'Avg Epoch Time']
    table_data = []
    
    for r in sorted(results, key=lambda x: (x['n_mels'], x['groups'])):
        table_data.append([
            r['n_mels'],
            r['groups'],
            f"{r['parameters']:,}",
            f"{r['test_accuracy']:.4f}",
            f"{np.mean(r['epoch_times']):.1f}s"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary table")

def plot_comparison_mel(mel_comparison_path, output_dir):
    """Plot mel spectrogram comparison"""
    # This plot was already generated
    pass

def main():
    results_file = 'results/all_results.json'
    output_dir = 'results'
    
    results = load_results(results_file)
    print(f"Loaded {len(results)} experiment results")
    
    plot_mel_comparison(results, output_dir)
    plot_groups_comparison(results, output_dir)
    plot_summary_table(results, output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()