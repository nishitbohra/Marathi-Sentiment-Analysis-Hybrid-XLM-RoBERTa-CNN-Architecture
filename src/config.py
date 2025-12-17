"""
Configuration module for Marathi Sentiment Analysis project.

This module contains all hyperparameters and settings for training
the hybrid XLM-RoBERTa + CNN sentiment analysis model.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Any
from pathlib import Path


@dataclass
class Config:
    """
    Configuration class for Marathi Sentiment Analysis project.
    
    Contains all hyperparameters, paths, and settings for:
    - Dataset configuration
    - Model architecture
    - Training parameters
    - Traditional ML baselines
    - Deep learning baselines
    
    Attributes:
        data_dir: Directory containing dataset CSV files
        train_file: Training dataset filename
        test_file: Test dataset filename
        val_file: Validation dataset filename
        plm_name: Pre-trained language model name
        max_seq_length: Maximum sequence length for tokenization
        batch_size: Batch size for training
        learning_rate_hybrid: Learning rate for hybrid model
        learning_rate_plm: Learning rate for PLM fine-tuning
        num_epochs: Maximum number of training epochs
        early_stopping_patience: Patience for early stopping
        plm_hidden_size: Hidden size of PLM (768 for XLM-RoBERTa base)
        cnn_out_channels: Output channels for CNN layer
        cnn_kernel_size: Kernel size for CNN
        dense_hidden_size: Hidden size for dense layers
        dropout_rate: Dropout rate
        num_classes: Number of sentiment classes
        warmup_steps: Warmup steps for scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        tfidf_max_features: Maximum features for TF-IDF vectorizer
        embedding_dim: Embedding dimension for DL baselines
        lstm_hidden_size: Hidden size for LSTM models
        vocab_size: Vocabulary size for DL baselines
        results_dir: Directory for saving results
        figures_dir: Directory for saving figures
        models_dir: Directory for saving models
        label_map: Mapping from original labels to PyTorch labels
        label_names: Names of sentiment classes
        device: Device for training (CPU/GPU)
    """
    
    # ==================== Dataset Configuration ====================
    data_dir: str = "data"
    train_file: str = "MahaSent_All_Train.csv"
    test_file: str = "MahaSent_All_Test.csv"
    val_file: str = "MahaSent_All_Val.csv"
    
    # ==================== Model Configuration ====================
    # Pre-trained Language Model
    plm_name: str = "xlm-roberta-base"
    max_seq_length: int = 256
    
    # Batch Size
    batch_size: int = 64
    
    # Learning Rates
    learning_rate_hybrid: float = 1e-4  # For hybrid model
    learning_rate_plm: float = 2e-5     # For PLM fine-tuning
    
    # Training Parameters
    num_epochs: int = 20
    early_stopping_patience: int = 3
    
    # ==================== Architecture Parameters ====================
    # PLM Configuration
    plm_hidden_size: int = 768  # XLM-RoBERTa base hidden size
    
    # CNN Configuration
    cnn_out_channels: int = 256
    cnn_kernel_size: int = 3
    
    # Classification Head
    dense_hidden_size: int = 512
    dropout_rate: float = 0.3
    num_classes: int = 3  # Negative, Neutral, Positive
    
    # ==================== Optimizer & Scheduler ====================
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ==================== Traditional ML Parameters ====================
    tfidf_max_features: int = 5000
    
    # ==================== Deep Learning Baseline Parameters ====================
    embedding_dim: int = 300
    lstm_hidden_size: int = 128
    vocab_size: int = 10000
    
    # ==================== Paths ====================
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    models_dir: str = "results/models"
    
    # ==================== Label Configuration ====================
    # CRITICAL: Dataset has labels {-1, 0, 1}
    # Convert to {0, 1, 2} for PyTorch CrossEntropyLoss
    label_map: Dict[int, int] = field(default_factory=lambda: {
        -1: 0,  # Negative → 0
        0: 1,   # Neutral → 1
        1: 2    # Positive → 2
    })
    
    label_names: List[str] = field(default_factory=lambda: [
        'Negative',  # Class 0
        'Neutral',   # Class 1
        'Positive'   # Class 2
    ])
    
    # Reverse mapping for interpretation
    reverse_label_map: Dict[int, int] = field(default_factory=lambda: {
        0: -1,  # 0 → Negative
        1: 0,   # 1 → Neutral
        2: 1    # 2 → Positive
    })
    
    # ==================== Device Configuration ====================
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    def __post_init__(self):
        """
        Post-initialization processing.
        Creates necessary directories and validates configuration.
        """
        # Create directories
        Path(self.results_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figures_dir).mkdir(exist_ok=True, parents=True)
        Path(self.models_dir).mkdir(exist_ok=True, parents=True)
        
        # Validate configuration
        assert self.num_classes == 3, "Number of classes must be 3 for this dataset"
        assert self.max_seq_length > 0, "Maximum sequence length must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 < self.dropout_rate < 1, "Dropout rate must be between 0 and 1"
        assert self.learning_rate_hybrid > 0, "Learning rate must be positive"
        assert self.learning_rate_plm > 0, "Learning rate must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            # Dataset
            'data_dir': self.data_dir,
            'train_file': self.train_file,
            'test_file': self.test_file,
            'val_file': self.val_file,
            
            # Model
            'plm_name': self.plm_name,
            'max_seq_length': self.max_seq_length,
            'batch_size': self.batch_size,
            'learning_rate_hybrid': self.learning_rate_hybrid,
            'learning_rate_plm': self.learning_rate_plm,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            
            # Architecture
            'plm_hidden_size': self.plm_hidden_size,
            'cnn_out_channels': self.cnn_out_channels,
            'cnn_kernel_size': self.cnn_kernel_size,
            'dense_hidden_size': self.dense_hidden_size,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes,
            
            # Training
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            
            # Traditional ML
            'tfidf_max_features': self.tfidf_max_features,
            
            # DL Baselines
            'embedding_dim': self.embedding_dim,
            'lstm_hidden_size': self.lstm_hidden_size,
            'vocab_size': self.vocab_size,
            
            # Device
            'device': str(self.device),
        }
    
    def print_config(self) -> None:
        """Print configuration in a formatted manner."""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        
        print("\n📁 Dataset Configuration:")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Train file: {self.train_file}")
        print(f"   Test file: {self.test_file}")
        print(f"   Val file: {self.val_file}")
        
        print("\n🤖 Model Configuration:")
        print(f"   PLM: {self.plm_name}")
        print(f"   Max sequence length: {self.max_seq_length}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Number of classes: {self.num_classes}")
        
        print("\n🏗️ Architecture:")
        print(f"   PLM hidden size: {self.plm_hidden_size}")
        print(f"   CNN out channels: {self.cnn_out_channels}")
        print(f"   CNN kernel size: {self.cnn_kernel_size}")
        print(f"   Dense hidden size: {self.dense_hidden_size}")
        print(f"   Dropout rate: {self.dropout_rate}")
        
        print("\n📚 Training Parameters:")
        print(f"   Learning rate (Hybrid): {self.learning_rate_hybrid}")
        print(f"   Learning rate (PLM): {self.learning_rate_plm}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Early stopping patience: {self.early_stopping_patience}")
        print(f"   Warmup steps: {self.warmup_steps}")
        print(f"   Weight decay: {self.weight_decay}")
        
        print("\n🔢 Label Mapping:")
        print(f"   Original → PyTorch: {self.label_map}")
        print(f"   Label names: {self.label_names}")
        
        print("\n💻 Device:")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print("\n📂 Output Directories:")
        print(f"   Results: {self.results_dir}")
        print(f"   Figures: {self.figures_dir}")
        print(f"   Models: {self.models_dir}")
        
        print("\n" + "=" * 70)


# Create a default configuration instance
def get_config() -> Config:
    """
    Get default configuration instance.
    
    Returns:
        Config instance with default parameters
    """
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    config.print_config()
    
    # Test to_dict method
    print("\n📋 Configuration Dictionary:")
    import json
    print(json.dumps(config.to_dict(), indent=2))
