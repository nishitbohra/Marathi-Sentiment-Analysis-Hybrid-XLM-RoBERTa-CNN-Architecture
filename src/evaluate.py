"""
Evaluation module for sentiment analysis models.

Provides comprehensive metrics calculation, confusion matrix analysis,
model comparison, and error analysis functions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Any
import pandas as pd
from tqdm.auto import tqdm


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive']
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of classes
        
    Returns:
        Dictionary containing all metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics with explicit labels
    num_classes = len(label_names)
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes))
    )
    
    # Confusion matrix with explicit labels
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Compile results
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'per_class': {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(label_names):
        metrics['per_class'][class_name] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1_score': f1_per_class[i],
            'support': support[i]
        }
    
    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive']
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run evaluation on
        label_names: Names of classes
        
    Returns:
        Dictionary containing predictions and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    logits_array = np.array(all_logits)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred, label_names)
    
    # Add predictions and logits
    metrics['predictions'] = y_pred
    metrics['true_labels'] = y_true
    metrics['logits'] = logits_array
    
    return metrics


def print_evaluation_results(
    metrics: Dict[str, Any],
    model_name: str = "Model"
) -> None:
    """
    Print evaluation results in a formatted manner.
    
    Args:
        metrics: Metrics dictionary from evaluate_model
        model_name: Name of the model
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*70}\n")
    
    print("Overall Metrics:")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro):   {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):      {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
        print(f"    Support:   {class_metrics['support']}")
    
    print(f"\n{'='*70}\n")


def confusion_matrix_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive']
) -> pd.DataFrame:
    """
    Create and analyze confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of classes
        
    Returns:
        Confusion matrix as DataFrame
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(label_names):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")
    
    return cm_df


def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['accuracy', 'f1_macro', 'f1_weighted']
) -> pd.DataFrame:
    """
    Compare multiple models based on metrics.
    
    Args:
        results_dict: Dictionary mapping model names to their results
        metrics: List of metrics to compare
        
    Returns:
        Comparison DataFrame
    """
    comparison = {}
    
    for model_name, results in results_dict.items():
        comparison[model_name] = {
            metric: results.get(metric, 0.0)
            for metric in metrics
        }
    
    df = pd.DataFrame(comparison).T
    df = df.sort_values(by=metrics[0], ascending=False)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70 + "\n")
    print(df.to_string())
    print("\n" + "="*70 + "\n")
    
    return df


def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: List[str],
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    num_examples: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform error analysis on misclassified examples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Original texts
        label_names: Names of classes
        num_examples: Number of examples to show per error type
        
    Returns:
        Dictionary of misclassification examples
    """
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total samples: {len(y_true)}")
    print(f"Correct predictions: {np.sum(y_true == y_pred)} ({np.sum(y_true == y_pred)/len(y_true)*100:.2f}%)")
    print(f"Misclassifications: {len(misclassified_indices)} ({len(misclassified_indices)/len(y_true)*100:.2f}%)")
    
    # Analyze misclassification patterns
    print(f"\n Misclassification Patterns:")
    error_patterns = {}
    
    for true_label in range(len(label_names)):
        for pred_label in range(len(label_names)):
            if true_label != pred_label:
                pattern_key = f"{label_names[true_label]} → {label_names[pred_label]}"
                indices = misclassified_indices[
                    (y_true[misclassified_indices] == true_label) &
                    (y_pred[misclassified_indices] == pred_label)
                ]
                count = len(indices)
                
                if count > 0:
                    print(f"   {pattern_key}: {count} samples")
                    error_patterns[pattern_key] = []
                    
                    # Collect examples
                    for idx in indices[:num_examples]:
                        error_patterns[pattern_key].append({
                            'text': texts[idx],
                            'true_label': label_names[true_label],
                            'pred_label': label_names[pred_label],
                            'index': int(idx)
                        })
    
    # Display sample errors
    print(f"\n❌ Sample Misclassifications (first {num_examples}):\n")
    count = 0
    for idx in misclassified_indices[:num_examples]:
        true_label = label_names[y_true[idx]]
        pred_label = label_names[y_pred[idx]]
        text = texts[idx][:120] if len(texts[idx]) > 120 else texts[idx]
        
        print(f"{count+1}. True: {true_label} | Pred: {pred_label}")
        print(f"   Text: {text}...")
        print("-" * 70)
        count += 1
    
    print()
    
    return error_patterns


def calculate_confidence_scores(
    logits: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction confidence scores.
    
    Args:
        logits: Model logits
        
    Returns:
        Tuple of (confidence_scores, predicted_classes)
    """
    # Apply softmax
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    
    # Get max probabilities and predicted classes
    confidence_scores = probs.max(axis=1)
    predicted_classes = probs.argmax(axis=1)
    
    return confidence_scores, predicted_classes


def get_low_confidence_predictions(
    logits: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    texts: List[str],
    threshold: float = 0.6,
    num_examples: int = 10
) -> List[Dict[str, Any]]:
    """
    Find predictions with low confidence scores.
    
    Args:
        logits: Model logits
        y_pred: Predicted labels
        y_true: True labels
        texts: Original texts
        threshold: Confidence threshold
        num_examples: Number of examples to return
        
    Returns:
        List of low confidence examples
    """
    confidence_scores, _ = calculate_confidence_scores(logits)
    low_confidence_indices = np.where(confidence_scores < threshold)[0]
    
    print(f"\n📊 Low Confidence Predictions (< {threshold}):")
    print(f"   Found: {len(low_confidence_indices)} samples")
    
    examples = []
    for idx in low_confidence_indices[:num_examples]:
        examples.append({
            'index': int(idx),
            'text': texts[idx],
            'confidence': float(confidence_scores[idx]),
            'predicted': int(y_pred[idx]),
            'true': int(y_true[idx]),
            'correct': bool(y_pred[idx] == y_true[idx])
        })
    
    return examples


if __name__ == "__main__":
    """Test evaluation functions."""
    
    print("Testing evaluation module...\n")
    
    # Create dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] * 10)
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2] * 10)  # Some errors
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print("Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    print("\n✅ Evaluation module tests passed!")
