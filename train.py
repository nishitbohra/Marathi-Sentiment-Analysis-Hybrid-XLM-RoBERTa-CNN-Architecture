"""
Streamlined training script - Train the Marathi Sentiment Analysis model.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import time
from src.config import Config
from src.data_loader import get_dataloaders_from_config
from src.models.hybrid_model import HybridSentimentModel
from src.train import train_epoch, validate, setup_training
from src.evaluate import evaluate_model
from src.visualize import plot_training_history, plot_confusion_matrix
from src.utils.logger import ExperimentLogger
import os

print("="*70)
print("MARATHI SENTIMENT ANALYSIS - TRAINING")
print("="*70)

# 1. Configuration
print("\n[1/6] Configuration...")
config = Config()
config.num_epochs = 10  # Reduced for faster training
print(f"✅ Device: {config.device}, Epochs: {config.num_epochs}, Batch: {config.batch_size}")

# 2. Setup logger
logger = ExperimentLogger("hybrid_training")
logger.log_config(config.to_dict())

# 3. Load data
print("\n[2/6] Loading data...")
tokenizer = AutoTokenizer.from_pretrained(config.plm_name)
train_loader, val_loader, test_loader = get_dataloaders_from_config(config, tokenizer)
print(f"✅ Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

# 4. Initialize model
print("\n[3/6] Initializing model...")
model = HybridSentimentModel(config).to(config.device)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {total_params:,} parameters (~{total_params*4/(1024**2):.0f}MB)")

# 5. Setup training
print("\n[4/6] Setup training...")
criterion, optimizer, scheduler = setup_training(model, train_loader, config)
print(f"✅ Optimizer: AdamW, LR: {config.learning_rate_hybrid}")

# 6. Train
print("\n[5/6] Training...")
print("="*70)

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
epochs_no_improve = 0

for epoch in range(1, config.num_epochs + 1):
    start_time = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        max_grad_norm=config.max_grad_norm,
        scheduler=scheduler
    )
    
    # Validate  
    val_loss, val_acc, _, _ = validate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=config.device
    )
    
    # Record
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    epoch_time = time.time() - start_time
    
    # Print
    print(f"Epoch {epoch}/{config.num_epochs} | "
          f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
          f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} | "
          f"Time={epoch_time:.1f}s")
    
    # Log
    logger.log_metrics({'loss': train_loss, 'accuracy': train_acc}, epoch, 'train')
    logger.log_metrics({'loss': val_loss, 'accuracy': val_acc}, epoch, 'val')
    
    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, os.path.join(config.models_dir, 'best_model.pt'))
        
        print(f"  ✓ New best! Saved model (Val Acc: {val_acc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break

print("\n" + "="*70)
print(f"✅ Training complete! Best Val Acc: {best_val_acc:.4f}")

# 7. Visualize
print("\n[6/6] Evaluation...")
Path(config.figures_dir).mkdir(parents=True, exist_ok=True)
plot_training_history(history, save_path=f"{config.figures_dir}/training_history.png")

# Load best model
checkpoint = torch.load(os.path.join(config.models_dir, 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])

# Test
results = evaluate_model(model, test_loader, config.device)
print(f"\n{'='*70}")
print("FINAL TEST RESULTS")
print('='*70)
print(f"Test Accuracy:     {results['accuracy']:.4f}")
print(f"Test F1 (macro):   {results['f1_macro']:.4f}")
print(f"Test Precision:    {results['precision_macro']:.4f}")
print(f"Test Recall:       {results['recall_macro']:.4f}")

print(f"\nPer-Class:")
for name in config.label_names:
    m = results['per_class'][name]
    print(f"  {name:8s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")

plot_confusion_matrix(results['confusion_matrix'], config.label_names,
                     save_path=f"{config.figures_dir}/confusion_matrix.png")

logger.log_final_results({'test_acc': results['accuracy'], 'test_f1': results['f1_macro']})

print(f"\n{'='*70}")
print("✅ PROJECT COMPLETE!")
print(f"{'='*70}")
print(f"Results: {config.results_dir}")
print("🎉 Marathi Sentiment Analysis finished!")
