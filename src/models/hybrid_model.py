"""
Hybrid sentiment analysis model combining XLM-RoBERTa and CNN.

This module implements the main hybrid architecture for Marathi sentiment analysis,
combining contextual features from XLM-RoBERTa with local pattern extraction via CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel
from typing import Tuple


class HybridSentimentModel(nn.Module):
    """
    Hybrid XLM-RoBERTa + CNN model for Marathi sentiment analysis.
    
    Architecture:
        1. XLM-RoBERTa base → Contextual embeddings (batch, seq_len, 768)
        2. Mean Pooling (attention-masked) → (batch, 768)
        3. Conv1D → Local pattern extraction (batch, 256)
        4. Concatenate [mean_pooled + cnn] → (batch, 1024)
        5. Dense layers (1024 → 512 → 3) with Dropout(0.3)
        6. Output logits (NO softmax, for CrossEntropyLoss)
    
    Args:
        config: Configuration object with model parameters
        
    Attributes:
        plm: Pre-trained XLM-RoBERTa model
        cnn: 1D Convolutional layer
        dropout: Dropout layer
        fc1: First dense layer
        fc2: Output layer
        relu: ReLU activation
    """
    
    def __init__(self, config):
        super(HybridSentimentModel, self).__init__()
        
        self.config = config
        
        # Load pre-trained XLM-RoBERTa
        self.plm = XLMRobertaModel.from_pretrained(config.plm_name)
        
        # CNN for local pattern extraction
        self.cnn = nn.Conv1d(
            in_channels=config.plm_hidden_size,
            out_channels=config.cnn_out_channels,
            kernel_size=config.cnn_kernel_size,
            padding='same'
        )
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(
            config.plm_hidden_size + config.cnn_out_channels,
            config.dense_hidden_size
        )
        self.fc2 = nn.Linear(config.dense_hidden_size, config.num_classes)
        
        self.relu = nn.ReLU()
    
    def mean_pooling(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling with attention mask.
        
        Args:
            token_embeddings: Token embeddings (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        # Expand attention mask to match embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # Sum mask values and avoid division by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Calculate mean
        return sum_embeddings / sum_mask
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes) - NO softmax applied
        """
        # Get PLM embeddings
        plm_output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = plm_output.last_hidden_state  # (batch, seq_len, 768)
        
        # Mean pooling for global representation
        pooled_output = self.mean_pooling(token_embeddings, attention_mask)  # (batch, 768)
        
        # CNN for local patterns
        # Transpose for Conv1D: (batch, seq_len, 768) → (batch, 768, seq_len)
        cnn_input = token_embeddings.permute(0, 2, 1)
        cnn_output = self.cnn(cnn_input)  # (batch, 256, seq_len)
        cnn_output = F.relu(cnn_output)
        
        # Max pooling over sequence dimension
        cnn_pooled = F.max_pool1d(cnn_output, kernel_size=cnn_output.size(2))  # (batch, 256, 1)
        cnn_pooled = cnn_pooled.squeeze(2)  # (batch, 256)
        
        # Concatenate features
        combined = torch.cat([pooled_output, cnn_pooled], dim=1)  # (batch, 1024)
        
        # Classification layers
        x = self.dropout(combined)
        x = self.fc1(x)  # (batch, 512)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (batch, 3)
        
        # Return logits (NO softmax - CrossEntropyLoss expects raw logits)
        return logits
    
    def freeze_plm(self) -> None:
        """Freeze the PLM parameters (for feature extraction mode)."""
        for param in self.plm.parameters():
            param.requires_grad = False
    
    def unfreeze_plm(self) -> None:
        """Unfreeze the PLM parameters (for fine-tuning)."""
        for param in self.plm.parameters():
            param.requires_grad = True
    
    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    """Test the hybrid model."""
    from src.config import get_config
    
    print("Testing HybridSentimentModel...\n")
    
    # Get configuration
    config = get_config()
    
    # Initialize model
    model = HybridSentimentModel(config)
    print(f"✅ Model initialized")
    print(f"   Total parameters: {model.get_num_total_params():,}")
    print(f"   Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Create dummy input
    batch_size = 4
    seq_length = 256
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    print(f"\n🔍 Testing forward pass:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected shape: ({batch_size}, {config.num_classes})")
    
    assert logits.shape == (batch_size, config.num_classes), "Output shape mismatch!"
    
    print(f"\n✅ All tests passed!")
