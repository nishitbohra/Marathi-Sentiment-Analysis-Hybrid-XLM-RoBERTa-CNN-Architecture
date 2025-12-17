"""
Visualization module for sentiment analysis results.

Provides plotting functions for training history, confusion matrices,
model comparisons, and error analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix array
        label_names: Names of classes
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        cmap: Colormap name
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1_macro'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to metrics
        metrics: List of metrics to compare
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    model_names = list(results_dict.keys())
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    for i, metric in enumerate(metrics):
        values = [results_dict[name][metric] for name in model_names]
        offset = width * (i - len(metrics)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(),
                     alpha=0.8, color=colors[i % len(colors)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to: {save_path}")
    
    plt.show()


def plot_per_class_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot per-class performance metrics.
    
    Args:
        metrics_dict: Dictionary with per-class metrics
        label_names: Names of classes
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Prepare data
    data = {
        'Precision': [metrics_dict[name]['precision'] for name in label_names],
        'Recall': [metrics_dict[name]['recall'] for name in label_names],
        'F1-Score': [metrics_dict[name]['f1_score'] for name in label_names]
    }
    
    df = pd.DataFrame(data, index=label_names)
    
    ax = df.plot(kind='bar', figsize=figsize, width=0.8, alpha=0.8)
    ax.set_xlabel('Sentiment Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(label_names, rotation=0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics plot saved to: {save_path}")
    
    plt.show()


def plot_text_length_distribution(
    correct_lengths: np.ndarray,
    misclassified_lengths: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot text length distribution for correct vs misclassified predictions.
    
    Args:
        correct_lengths: Text lengths of correctly classified samples
        misclassified_lengths: Text lengths of misclassified samples
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.hist(correct_lengths, bins=50, alpha=0.6, label='Correct',
            color='green', edgecolor='black')
    plt.hist(misclassified_lengths, bins=50, alpha=0.6, label='Misclassified',
            color='red', edgecolor='black')
    
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Text Length Distribution: Correct vs Misclassified',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Text length distribution plot saved to: {save_path}")
    
    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    title: str = 'Class Distribution',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot class distribution as bar chart.
    
    Args:
        labels: Array of labels
        label_names: Names of classes
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(unique)), counts, alpha=0.8, edgecolor='black')
    
    # Color bars
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(unique)), [label_names[i] for i in unique])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, count,
                f'{count}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Class distribution plot saved to: {save_path}")
    
    plt.show()


def create_results_summary_figure(
    results_dict: Dict[str, Any],
    save_dir: str = "results/figures"
) -> None:
    """
    Create a comprehensive summary figure with multiple subplots.
    
    Args:
        results_dict: Dictionary containing all results and metrics
        save_dir: Directory to save figures
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Model comparison
    ax1 = fig.add_subplot(gs[0, :])
    # ... implementation
    
    # Plot 2: Confusion matrix
    ax2 = fig.add_subplot(gs[1, 0])
    # ... implementation
    
    # Plot 3: Per-class metrics
    ax3 = fig.add_subplot(gs[1, 1])
    # ... implementation
    
    # Plot 4: Training history
    ax4 = fig.add_subplot(gs[2, :])
    # ... implementation
    
    plt.suptitle('Sentiment Analysis Results Summary',
                fontsize=16, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_dir, 'results_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Results summary figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    """Test visualization functions."""
    
    print("Testing visualization module...\n")
    
    # Test training history plot
    history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35],
        'train_acc': [0.6, 0.7, 0.75, 0.8, 0.83],
        'val_loss': [0.7, 0.55, 0.48, 0.42, 0.40],
        'val_acc': [0.65, 0.72, 0.76, 0.79, 0.81]
    }
    
    print("Plotting training history...")
    plot_training_history(history)
    
    # Test confusion matrix
    cm = np.array([[80, 10, 10], [5, 85, 10], [8, 7, 85]])
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(cm)
    
    print("\n✅ Visualization module tests passed!")
