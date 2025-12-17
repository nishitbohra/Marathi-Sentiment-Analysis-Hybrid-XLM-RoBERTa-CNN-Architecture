"""
Pre-trained Language Model (PLM) fine-tuning wrapper.

This module provides a wrapper for fine-tuning XLM-RoBERTa and other
pre-trained language models for sentiment classification.
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig


class PLMClassifier(nn.Module):
    """
    Pre-trained Language Model with classification head.
    
    Simple wrapper that adds a linear classification layer on top of PLM.
    Uses CLS token representation for classification.
    
    Args:
        plm_name: Name of pre-trained model (e.g., 'xlm-roberta-base')
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_plm: If True, freeze PLM parameters (feature extraction mode)
    """
    
    def __init__(
        self,
        plm_name: str = "xlm-roberta-base",
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_plm: bool = False
    ):
        super(PLMClassifier, self).__init__()
        
        self.plm_name = plm_name
        self.num_classes = num_classes
        
        # Load pre-trained model
        self.plm = XLMRobertaModel.from_pretrained(plm_name)
        
        # Get hidden size from config
        self.hidden_size = self.plm.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Freeze PLM if specified
        if freeze_plm:
            self.freeze_plm()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get PLM outputs
        outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Classification
        output = self.dropout(cls_output)
        logits = self.classifier(output)  # (batch_size, num_classes)
        
        return logits
    
    def freeze_plm(self) -> None:
        """Freeze all PLM parameters."""
        for param in self.plm.parameters():
            param.requires_grad = False
        print("🔒 PLM parameters frozen")
    
    def unfreeze_plm(self) -> None:
        """Unfreeze all PLM parameters."""
        for param in self.plm.parameters():
            param.requires_grad = True
        print("🔓 PLM parameters unfrozen")
    
    def freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first N layers of the PLM.
        
        Args:
            num_layers: Number of layers to freeze
        """
        # Freeze embeddings
        for param in self.plm.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i in range(num_layers):
            for param in self.plm.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"🔒 Froze first {num_layers} layers of PLM")
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class PLMWithMeanPooling(nn.Module):
    """
    PLM with mean pooling instead of CLS token.
    
    Args:
        plm_name: Name of pre-trained model
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        plm_name: str = "xlm-roberta-base",
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(PLMWithMeanPooling, self).__init__()
        
        self.plm = XLMRobertaModel.from_pretrained(plm_name)
        self.hidden_size = self.plm.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
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
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get PLM outputs
        outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        pooled_output = self.mean_pooling(token_embeddings, attention_mask)
        
        # Classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits


def get_plm_model(
    plm_name: str = "xlm-roberta-base",
    num_classes: int = 3,
    dropout: float = 0.3,
    pooling: str = "cls",
    freeze_plm: bool = False
) -> nn.Module:
    """
    Factory function to create PLM models.
    
    Args:
        plm_name: Name of pre-trained model
        num_classes: Number of output classes
        dropout: Dropout rate
        pooling: Pooling strategy ('cls' or 'mean')
        freeze_plm: Whether to freeze PLM parameters
        
    Returns:
        Initialized PLM model
    """
    if pooling == "cls":
        return PLMClassifier(plm_name, num_classes, dropout, freeze_plm)
    elif pooling == "mean":
        return PLMWithMeanPooling(plm_name, num_classes, dropout)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")


if __name__ == "__main__":
    """Test PLM models."""
    
    print("Testing PLM Models...\n")
    
    # Test parameters
    batch_size = 4
    seq_len = 256
    num_classes = 3
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}\n")
    
    # Test CLS pooling
    print("Testing PLM with CLS pooling:")
    model_cls = get_plm_model(pooling="cls")
    print(f"  Total parameters: {model_cls.get_num_total_params():,}")
    print(f"  Trainable parameters: {model_cls.get_num_trainable_params():,}")
    
    with torch.no_grad():
        logits_cls = model_cls(input_ids, attention_mask)
    print(f"  Output shape: {logits_cls.shape}")
    print(f"  ✅ Test passed\n")
    
    # Test mean pooling
    print("Testing PLM with mean pooling:")
    model_mean = get_plm_model(pooling="mean")
    
    with torch.no_grad():
        logits_mean = model_mean(input_ids, attention_mask)
    print(f"  Output shape: {logits_mean.shape}")
    print(f"  ✅ Test passed\n")
    
    # Test freezing
    print("Testing parameter freezing:")
    model_frozen = get_plm_model(freeze_plm=True)
    print(f"  Trainable parameters: {model_frozen.get_num_trainable_params():,}")
    print(f"  ✅ Test passed\n")
    
    print("✅ All PLM model tests passed!")
