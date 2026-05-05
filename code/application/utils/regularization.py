"""
Regularization utilities for neural network training.

This module provides functions to apply various regularization techniques:
- L1 regularization (L1 penalty on weights)
- L2 regularization (weight decay via optimizer)
- Max-norm constraint (weight clipping)
- Sparsity penalties (L1-like penalty encouraging sparse weights)

Note: Dropout is handled separately in model architecture and is not managed here.
Dropout can be used simultaneously with other regularization methods.
"""

import torch
import torch.nn as nn
from torch import optim

def apply_regularization_loss(loss, model, r_method, r_weight):
    """
    Apply regularization penalties to the loss function.
    Args:
        loss: The original loss tensor
        model: The neural network model
        r_method: Regularization method ('l1', 'sparsity', 'l2', 'maxnorm', 'none')
        r_weight: Regularization weight/strength
    Returns:
        Modified loss tensor with regularization penalties added
    """
    if r_method is None or r_method == "none" or r_method == "maxnorm" or r_method == "l2":
        # L2 is handled via optimizer weight_decay, max-norm is applied after optimizer step
        return loss
    if r_method == "l1" or r_method == "sparsity":
        # Apply L1/sparsity penalty
        l1_reg = 0
        for param in model.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
        loss = loss + r_weight * l1_reg
    return loss
def apply_max_norm_constraint(model, r_weight):
    """
    Apply max-norm constraint to model weights.
    Args:
        model: The neural network model
        r_weight: Maximum norm value for weight clipping
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.data.norm(2)
                if param_norm > r_weight:
                    param.data.mul_(r_weight / param_norm)
def create_optimizer_with_reg(model, optimizer_function, lr, r_method, r_weight):
    """
    Create optimizer with appropriate regularization settings.
    Args:
        model: The neural network model
        optimizer_function: Optimizer type ('adam', 'sgd', 'rmsprop')
        lr: Learning rate
        r_method: Regularization method ('l1', 'l2', 'sparsity', 'maxnorm', 'none')
        r_weight: Regularization weight (used for L2 weight_decay)
    Returns:
        Optimizer instance
    """
    # For L2 regularization, use weight_decay parameter in optimizer
    # For other methods, weight_decay should be 0
    if r_method == "l2":
        weight_decay = r_weight
    else:
        weight_decay = 0.0
    if optimizer_function == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_function == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)
    elif optimizer_function == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_function == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Default to Adam if unknown optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
def get_regularization_info(r_method, r_weight, dropout_rate):
    """
    Get regularization information for logging.
    Args:
        r_method: Regularization method
        r_weight: Regularization weight
        dropout_rate: Dropout rate
    Returns:
        Dictionary with regularization information
    """
    info = {
        'r_method': r_method if r_method else 'none',
        'r_weight': r_weight if r_weight else 0.0,
        'dropout_rate': dropout_rate if dropout_rate else 0.0
    }
    return info
