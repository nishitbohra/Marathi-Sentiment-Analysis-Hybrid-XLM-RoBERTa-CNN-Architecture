"""
Training module for sentiment analysis models.

Provides functions for training, validation, and managing the training loop
with early stopping, checkpointing, and learning rate scheduling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from transformers import get_scheduler
from typing import Tuple, Dict, Any, Optional, List
from tqdm.auto import tqdm
import time
import os
from pathlib import Path


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    scheduler: Optional[Any] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Scheduler step (if provided)
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        current_acc = correct_predictions / total_samples
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        data_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy, predictions, true_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Get predictions
            _, preds = torch.max(logits, dim=1)
            
            # Store predictions and labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = (all_preds == all_labels).float().mean().item()
    
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = 20,
    early_stopping_patience: int = 3,
    scheduler: Optional[Any] = None,
    save_dir: str = "results/models",
    model_name: str = "model",
    max_grad_norm: float = 1.0
) -> Dict[str, List[float]]:
    """
    Train model with early stopping and checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        scheduler: Learning rate scheduler
        save_dir: Directory to save models
        model_name: Base name for saved models
        max_grad_norm: Maximum gradient norm
        
    Returns:
        Training history dictionary
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Starting Training: {model_name}")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm, scheduler
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
                break
        
        print()
    
    print(f"{'='*70}")
    print("✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"   Model saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return history


def setup_training(
    model: nn.Module,
    train_loader: DataLoader,
    config: Any,
    learning_rate: Optional[float] = None
) -> Tuple[nn.Module, Optimizer, Any]:
    """
    Setup optimizer and scheduler for training.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        config: Configuration object
        learning_rate: Learning rate (uses config if None)
        
    Returns:
        Tuple of (criterion, optimizer, scheduler)
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if learning_rate is None:
        learning_rate = config.learning_rate_hybrid
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    return criterion, optimizer, scheduler


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', -1),
        'val_loss': checkpoint.get('val_loss', None),
        'val_acc': checkpoint.get('val_acc', None)
    }
    
    print(f"✅ Loaded checkpoint from epoch {info['epoch'] + 1}")
    if info['val_loss'] is not None:
        print(f"   Val Loss: {info['val_loss']:.4f} | Val Acc: {info['val_acc']:.4f}")
    
    return model, info


if __name__ == "__main__":
    """Test training functions."""
    
    print("Training module loaded successfully!")
    print("Use this module to train your models with early stopping and checkpointing.")
