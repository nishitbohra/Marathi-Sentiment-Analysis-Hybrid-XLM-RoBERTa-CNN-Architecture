"""
Deep Learning baseline models for sentiment analysis.

This module implements various deep learning architectures as baselines:
- LSTM
- BiLSTM
- CNN
- Multi-CNN (with multiple kernel sizes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LSTMModel(nn.Module):
    """
    LSTM model for sentiment classification.
    
    Architecture:
        Embedding → LSTM → Dense → Output
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_size: LSTM hidden size
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: (1, batch, hidden_size)
        
        # Use last hidden state
        output = hidden.squeeze(0)  # (batch, hidden_size)
        
        # Classification
        output = self.dropout(output)
        logits = self.fc(output)  # (batch, num_classes)
        
        return logits


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for sentiment classification.
    
    Architecture:
        Embedding → BiLSTM → Dense → Output
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_size: LSTM hidden size (output will be 2*hidden_size)
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(BiLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        
        # BiLSTM
        lstm_out, (hidden, _) = self.bilstm(embedded)
        # hidden: (2, batch, hidden_size) for bidirectional
        
        # Concatenate forward and backward hidden states
        hidden_forward = hidden[0]
        hidden_backward = hidden[1]
        output = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.dropout(output)
        logits = self.fc(output)  # (batch, num_classes)
        
        return logits


class CNNModel(nn.Module):
    """
    CNN model for sentiment classification.
    
    Architecture:
        Embedding → Conv1D → MaxPool → Dense → Output
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        num_filters: Number of convolutional filters
        kernel_size: Kernel size for convolution
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 128,
        kernel_size: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(CNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same'
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        
        # Transpose for Conv1D: (batch, seq_len, embedding_dim) → (batch, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Convolution
        conv_out = self.conv(embedded)  # (batch, num_filters, seq_len)
        conv_out = F.relu(conv_out)
        
        # Global max pooling
        pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch, num_filters, 1)
        pooled = pooled.squeeze(2)  # (batch, num_filters)
        
        # Classification
        output = self.dropout(pooled)
        logits = self.fc(output)  # (batch, num_classes)
        
        return logits


class MultiCNNModel(nn.Module):
    """
    Multi-kernel CNN model for sentiment classification.
    
    Uses multiple convolutional layers with different kernel sizes
    to capture patterns at different scales.
    
    Architecture:
        Embedding → Multiple Conv1D (different kernels) → Concatenate → Dense → Output
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        num_filters: Number of filters per kernel size
        kernel_sizes: List of kernel sizes to use
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 128,
        kernel_sizes: List[int] = [2, 3, 4, 5],
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(MultiCNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding='same'
            )
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        
        # Transpose for Conv1D
        embedded = embedded.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        
        # Apply each convolution and pool
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # (batch, num_filters, seq_len)
            conv_out = F.relu(conv_out)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all pooled outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * num_kernels)
        
        # Classification
        output = self.dropout(concatenated)
        logits = self.fc(output)  # (batch, num_classes)
        
        return logits


def get_model(
    model_type: str,
    vocab_size: int,
    embedding_dim: int = 300,
    hidden_size: int = 128,
    num_classes: int = 3,
    dropout: float = 0.3
) -> nn.Module:
    """
    Factory function to create deep learning models.
    
    Args:
        model_type: Type of model ('lstm', 'bilstm', 'cnn', 'multicnn')
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_size: Hidden size (for LSTM/BiLSTM) or num_filters (for CNN)
        num_classes: Number of output classes
        dropout: Dropout rate
        
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMModel(vocab_size, embedding_dim, hidden_size, num_classes, dropout)
    elif model_type == 'bilstm':
        return BiLSTMModel(vocab_size, embedding_dim, hidden_size, num_classes, dropout)
    elif model_type == 'cnn':
        return CNNModel(vocab_size, embedding_dim, hidden_size, 3, num_classes, dropout)
    elif model_type == 'multicnn':
        return MultiCNNModel(vocab_size, embedding_dim, hidden_size, [2, 3, 4, 5], num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    """Test deep learning models."""
    
    print("Testing Deep Learning Models...\n")
    
    # Test parameters
    batch_size = 4
    seq_len = 100
    vocab_size = 10000
    num_classes = 3
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test each model
    models = {
        'LSTM': get_model('lstm', vocab_size),
        'BiLSTM': get_model('bilstm', vocab_size),
        'CNN': get_model('cnn', vocab_size),
        'Multi-CNN': get_model('multicnn', vocab_size)
    }
    
    print(f"Input shape: {input_ids.shape}\n")
    
    for name, model in models.items():
        print(f"Testing {name}:")
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        print(f"  Output shape: {logits.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  ✅ Test passed\n")
    
    print("✅ All deep learning model tests passed!")
