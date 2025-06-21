import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import warnings

import torch.nn as nn
import torch.nn.functional as F
from ensemblenet import logger # Use your logger
from torch import Tensor
import math
import os
class MultiHeadAttentionModule(nn.Module):
    """
    True Multi-Head Attention Module for ensemble features with explicit Q, K, V Linear layers.
    Input: (batch_size, num_models, embed_dim)
    Output: (batch_size, embed_dim), attention weights (batch_size, num_heads, num_models, num_models)
    """
    def __init__(self, embed_dim, num_models=3, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_models = num_models
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        logger.info(f"Initialized MultiHeadAttentionModule with explicit QKV: embed_dim={embed_dim}, num_models={num_models}, num_heads={num_heads}")

    def forward(self, x):
        # x: (batch_size, num_models, embed_dim)
        B, N, E = x.shape
        H = self.num_heads
        D = self.head_dim

        # Linear projections
        Q = self.q_proj(x)  # (B, N, E)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: (B, N, H, D) -> (B, H, N, D)
        Q = Q.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, E)  # (B, N, E)

        # Output projection and residual connection + normalization
        out = self.out_proj(attn_output)
        out = self.norm(out + x)

        # Pool over num_models dimension (mean)
        return out, attn_weights


def get_channels_by_forward_pass(model, input_size=(1, 3, 224, 224)):
    """
    Get output channels by doing a forward pass with dummy input.
    This is more reliable than inspection.
    
    Args:
        model: The feature extractor model
        input_size: Input tensor size (batch, channels, height, width)
        
    Returns:
        int: Number of output channels
    """
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        try:
            output = model(dummy_input)
            return output.shape[1]  # Return channel dimension
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return None
        
class EnsembleNet(nn.Module):
    def __init__(
        self, 
        num_classes, 
        submodule_feature_extractors=None,
        embed_dim=512
    ):
        super(EnsembleNet, self).__init__()
        
        # Create default feature extractors if none provided
        if submodule_feature_extractors is None:
            submodule_feature_extractors = [
                models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features,
                models.mnasnet0_5(weights=models.MNASNet0_5_Weights.DEFAULT).layers,
                models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).features,
            ]
        
        self.submodule_feature_extractors = nn.ModuleList(submodule_feature_extractors)
        
        # Get channels for each feature extractor
        submodule_last_channels = [get_channels_by_forward_pass(model) for model in submodule_feature_extractors]
        print(f"Detected channels: {submodule_last_channels}")
        
        # Create simple projection layers (removed batch norm for stability)
        self.submodule_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_channel, embed_dim, 1, 1),
                nn.ReLU(inplace=True)
            ) for n_channel in submodule_last_channels
        ])
        
        self.num_extractors = len(submodule_feature_extractors)
        
        # Simple attention mechanism (similar to AttentionEnsembleCNN)
        self.attention = MultiHeadAttentionModule(embed_dim, self.num_extractors)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Simplified classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.backbone_names = [
            'mobilenet_v3_small',
            'mnasnet0_5', 
            'squeezenet1_1'
        ]
        
    def load_backbone_weights(self, weight_paths, strict=True):
        """
        Load pre-trained weights for backbone feature extractors.
        
        Args:
            weight_paths (list or dict): 
                - If list: paths to weight files in order of backbones
                - If dict: mapping of backbone_name -> weight_path
            strict (bool): Whether to strictly match state dict keys
        """
        if isinstance(weight_paths, dict):
            # Dictionary mapping backbone names to paths
            for i, backbone_name in enumerate(self.backbone_names):
                if backbone_name in weight_paths:
                    self._load_single_backbone_weights(i, weight_paths[backbone_name], strict)
        elif isinstance(weight_paths, (list, tuple)):
            # List of paths in order
            if len(weight_paths) != len(self.submodule_feature_extractors):
                raise ValueError(f"Expected {len(self.submodule_feature_extractors)} weight paths, got {len(weight_paths)}")
            
            for i, weight_path in enumerate(weight_paths):
                if weight_path is not None:  # Allow None to skip loading for specific backbone
                    self._load_single_backbone_weights(i, weight_path, strict)
        else:
            raise TypeError("weight_paths must be a list, tuple, or dictionary")
    
    def freeze_backbones(self, backbone_indices=None):
        """
        Freeze backbone feature extractors.
        
        Args:
            backbone_indices (list or None): 
                - If None: freeze all backbones
                - If list: freeze only specified backbone indices
        """
        if backbone_indices is None:
            backbone_indices = list(range(len(self.submodule_feature_extractors)))
        elif isinstance(backbone_indices, int):
            backbone_indices = [backbone_indices]
        
        for idx in backbone_indices:
            if 0 <= idx < len(self.submodule_feature_extractors):
                for param in self.submodule_feature_extractors[idx].parameters():
                    param.requires_grad = False
                logger.info(f"Frozen backbone {idx} ({self.backbone_names[idx]})")
            else:
                logger.warning(f"Invalid backbone index: {idx}")
    
    def unfreeze_backbones(self, backbone_indices=None):
        """
        Unfreeze backbone feature extractors.
        
        Args:
            backbone_indices (list or None): 
                - If None: unfreeze all backbones
                - If list: unfreeze only specified backbone indices
        """
        if backbone_indices is None:
            backbone_indices = list(range(len(self.submodule_feature_extractors)))
        elif isinstance(backbone_indices, int):
            backbone_indices = [backbone_indices]
        
        for idx in backbone_indices:
            if 0 <= idx < len(self.submodule_feature_extractors):
                for param in self.submodule_feature_extractors[idx].parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen backbone {idx} ({self.backbone_names[idx]})")
            else:
                logger.warning(f"Invalid backbone index: {idx}")

    def _load_single_backbone_weights(self, backbone_idx, weight_path, strict=True):
        """Load weights for a single backbone."""
        if not os.path.exists(weight_path):
            logger.warning(f"Weight file not found: {weight_path}")
            return
        
        try:
            # Load checkpoint with weights_only=False to handle all formats
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            logger.info(f"Loaded checkpoint from {weight_path}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            elif hasattr(checkpoint, 'state_dict'):
                # If checkpoint is a model object, extract its state_dict
                state_dict = checkpoint.state_dict()
                logger.info(f"Extracted state_dict from model object for backbone {backbone_idx}")
            else:
                # If checkpoint is directly a state dict or other format
                try:
                    # Try to use it as a state dict
                    state_dict = checkpoint
                    # Test if it has the expected dict interface
                    if not hasattr(state_dict, 'items'):
                        raise ValueError("Checkpoint is not in a recognized format")
                except Exception as e:
                    logger.error(f"Unrecognized checkpoint format for backbone {backbone_idx}: {type(checkpoint)}")
                    raise ValueError(f"Cannot extract state_dict from checkpoint of type {type(checkpoint)}")
            
            # Filter state dict to match backbone structure
            backbone_state_dict = {}
            backbone_prefix = f'submodule_feature_extractors.{backbone_idx}.'
            
            for key, value in state_dict.items():
                if key.startswith(backbone_prefix):
                    # Remove the prefix to match the backbone's local naming
                    new_key = key[len(backbone_prefix):]
                    backbone_state_dict[new_key] = value
                elif not key.startswith('submodule_feature_extractors.'):
                    # If the checkpoint is from a standalone model, use keys as-is
                    backbone_state_dict[key] = value
            
            # Load the weights
            missing_keys, unexpected_keys = self.submodule_feature_extractors[backbone_idx].load_state_dict(
                backbone_state_dict, strict=strict
            )
                        
            logger.info(f"Successfully loaded weights for backbone {backbone_idx} ({self.backbone_names[backbone_idx]}) from {weight_path}")
            
        except Exception as e:
            logger.error(f"Error loading weights for backbone {backbone_idx} from {weight_path}: {str(e)}")
            raise
    
    
    def forward(self, x):
        # Extract and project features from each backbone
        projected_features = []
        for extractor, projection in zip(self.submodule_feature_extractors, self.submodule_projections):
            feat = extractor(x)
            projected = projection(feat)
            
            # Global average pooling
            projected = self.global_pool(projected).squeeze(-1).squeeze(-1)  # [B, embed_dim]
            projected_features.append(projected)
        
        # Stack features: [B, num_extractors, embed_dim]
        features = torch.stack(projected_features, dim=1)  # [B, num_extractors, embed_dim]

        # Apply simple attention mechanism
        attended_features, attention_weights = self.attention(features)  # [B, embed_dim]
        
        # Classification
        output = self.classifier(attended_features).mean(dim=1)  # Average over batch
        output = F.log_softmax(output, dim=1)  # Log softmax for
        return output
