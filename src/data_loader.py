"""
Data loading module for Marathi Sentiment Analysis project.

This module provides PyTorch Dataset and DataLoader creation for the
MahaSent dataset with proper label mapping and tokenization.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class MarathiSentimentDataset(Dataset):
    """
    PyTorch Dataset for Marathi sentiment analysis.
    
    Handles tokenization and label mapping for the MahaSent dataset.
    Converts original labels {-1, 0, 1} to PyTorch-compatible {0, 1, 2}.
    
    Args:
        texts: List of preprocessed text strings
        labels: List of original labels (-1, 0, 1)
        tokenizer: XLMRobertaTokenizer instance
        max_length: Maximum sequence length for tokenization
        label_map: Dictionary mapping original labels to PyTorch labels
        
    Attributes:
        texts: List of input texts
        labels: List of mapped labels (0, 1, 2)
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: XLMRobertaTokenizer,
        max_length: int = 256,
        label_map: Optional[Dict[int, int]] = None
    ):
        """Initialize the dataset."""
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default label mapping: -1→0, 0→1, 1→2
        if label_map is None:
            label_map = {-1: 0, 0: 1, 1: 2}
        
        # Map labels
        self.labels = [label_map[label] for label in labels]
        
        # Validate
        assert len(self.texts) == len(self.labels), \
            f"Mismatch: {len(self.texts)} texts vs {len(self.labels)} labels"
        assert all(label in [0, 1, 2] for label in self.labels), \
            "All labels must be in {0, 1, 2} after mapping"
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs (torch.Tensor)
                - attention_mask: Attention mask (torch.Tensor)
                - label: Mapped label (torch.Tensor)
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_dataset_from_csv(
    csv_path: str,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[List[str], List[int]]:
    """
    Load texts and labels from CSV file.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (texts, labels)
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in CSV")
    if label_column not in df.columns:
        raise KeyError(f"Column '{label_column}' not found in CSV")
    
    # Extract texts and labels
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Convert texts to strings and handle NaN
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    
    return texts, labels


def get_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer: XLMRobertaTokenizer,
    max_length: int = 256,
    batch_size: int = 64,
    label_map: Optional[Dict[int, int]] = None,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        test_texts: Test texts
        test_labels: Test labels
        tokenizer: XLMRobertaTokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        label_map: Dictionary mapping original labels to PyTorch labels
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory (set True if using GPU)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MarathiSentimentDataset(
        train_texts, train_labels, tokenizer, max_length, label_map
    )
    val_dataset = MarathiSentimentDataset(
        val_texts, val_labels, tokenizer, max_length, label_map
    )
    test_dataset = MarathiSentimentDataset(
        test_texts, test_labels, tokenizer, max_length, label_map
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_dataloaders_from_config(config, tokenizer: XLMRobertaTokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders using configuration object.
    
    Args:
        config: Configuration object with dataset paths and parameters
        tokenizer: XLMRobertaTokenizer instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> from src.config import get_config
        >>> from transformers import XLMRobertaTokenizer
        >>> 
        >>> config = get_config()
        >>> tokenizer = XLMRobertaTokenizer.from_pretrained(config.plm_name)
        >>> train_loader, val_loader, test_loader = get_dataloaders_from_config(config, tokenizer)
    """
    # Load datasets from CSV files
    train_path = os.path.join(config.data_dir, config.train_file)
    val_path = os.path.join(config.data_dir, config.val_file)
    test_path = os.path.join(config.data_dir, config.test_file)
    
    train_texts, train_labels = load_dataset_from_csv(train_path)
    val_texts, val_labels = load_dataset_from_csv(val_path)
    test_texts, test_labels = load_dataset_from_csv(test_path)
    
    # Determine if GPU is available for pin_memory
    pin_memory = torch.cuda.is_available()
    
    # Create and return DataLoaders
    return get_dataloaders(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        batch_size=config.batch_size,
        label_map=config.label_map,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=pin_memory
    )


def print_dataset_info(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader
) -> None:
    """
    Print information about the DataLoaders.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    print("=" * 70)
    print("DATALOADER INFORMATION")
    print("=" * 70)
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   Train: {len(train_loader.dataset):,} samples")
    print(f"   Val:   {len(val_loader.dataset):,} samples")
    print(f"   Test:  {len(test_loader.dataset):,} samples")
    print(f"   Total: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):,} samples")
    
    print(f"\n📦 Batch Configuration:")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\n🔍 Sample Batch Shapes:")
    print(f"   Input IDs:      {sample_batch['input_ids'].shape}")
    print(f"   Attention Mask: {sample_batch['attention_mask'].shape}")
    print(f"   Labels:         {sample_batch['label'].shape}")
    
    print(f"\n✅ DataLoaders ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    """Test the data loading functionality."""
    from transformers import XLMRobertaTokenizer
    
    # Test with dummy data
    print("Testing MarathiSentimentDataset...\n")
    
    # Create dummy data
    texts = [
        "हे एक चांगले चित्रपट आहे",
        "मला आवडत नाही",
        "सामान्य आहे"
    ]
    labels = [1, -1, 0]  # Original labels
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    
    # Create dataset
    dataset = MarathiSentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=256
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Original labels: {labels}")
    print(f"Mapped labels: {dataset.labels}")
    
    # Test __getitem__
    print("\nSample item:")
    sample = dataset[0]
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Label: {sample['label'].item()}")
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"\nDataLoader batches: {len(loader)}")
    
    # Test batch
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    
    print("\n✅ All tests passed!")
