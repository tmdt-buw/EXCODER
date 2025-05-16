import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.classification_base import ClassificationLightningModule
from typing import List, Tuple, Optional
from params import DATASET_NAMES


class ShapeletLayer(nn.Module):
    """
    Shapelet layer for time series classification.
    
    This layer learns a set of shapelets (discriminative subsequences) and computes
    the minimum distance between each shapelet and all subsequences of the input time series.
    
    Args:
        in_channels (int): Number of input channels (features)
        num_shapelets (int): Total number of shapelets to learn
        shapelet_lengths (List[int]): List of lengths for each group of shapelets
        num_shapelet_per_length (List[int]): Number of shapelets for each length
    """
    def __init__(
        self,
        in_channels: int,
        num_shapelets: int,
        shapelet_lengths: List[int],
        num_shapelet_per_length: List[int],
    ):
        super().__init__()
        
        assert len(shapelet_lengths) == len(num_shapelet_per_length), "Each shapelet length must have a corresponding count"
        assert sum(num_shapelet_per_length) == num_shapelets, "Sum of shapelets per length must equal total shapelets"
        
        self.in_channels = in_channels
        self.num_shapelets = num_shapelets
        self.shapelet_lengths = shapelet_lengths
        self.num_shapelet_per_length = num_shapelet_per_length
        
        # Initialize the shapelets as learnable parameters
        self.shapelets = nn.ParameterList()
        
        start_idx = 0
        for length, count in zip(shapelet_lengths, num_shapelet_per_length):
            # Initialize shapelets with random values from a normal distribution
            shapelet_group = nn.Parameter(
                torch.randn(count, in_channels, length) * 0.01
            )
            self.shapelets.append(shapelet_group)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the minimum distance between each shapelet and subsequences of x.
        
        Args:
            x (torch.Tensor): Input time series [batch_size, in_channels, seq_length]
            
        Returns:
            torch.Tensor: Minimum distances for each shapelet [batch_size, num_shapelets]
        """
        batch_size = x.shape[0]
        seq_length = x.shape[2]
        
        # Prepare the output tensor
        distances = torch.zeros(batch_size, self.num_shapelets, device=x.device)
        
        shapelet_idx = 0
        for shapelet_group, length in zip(self.shapelets, self.shapelet_lengths):
            num_shapelets_in_group = shapelet_group.shape[0]
            
            # Compute distances for this group of shapelets
            for i in range(seq_length - length + 1):
                # Extract subsequence
                subsequence = x[:, :, i:i+length].unsqueeze(1)  # [batch_size, 1, channels, length]
                
                # Calculate Euclidean distance between subsequence and all shapelets in this group
                # Reshape to [batch_size, num_shapelets_in_group, channels, length]
                shapelet_expanded = shapelet_group.unsqueeze(0).expand(batch_size, -1, -1, -1)
                
                # Calculate squared Euclidean distance
                dist = torch.sum((subsequence - shapelet_expanded) ** 2, dim=(2, 3))
                
                # Update minimum distances
                if i == 0:
                    min_distances = dist
                else:
                    min_distances = torch.minimum(min_distances, dist)
            
            # Store minimum distances for this shapelet group
            distances[:, shapelet_idx:shapelet_idx+num_shapelets_in_group] = min_distances
            shapelet_idx += num_shapelets_in_group
        
        return distances


class FastShapeletClassifier(ClassificationLightningModule):
    """
    A fast shapelet-based classifier for time series data.
    
    This model uses learnable shapelets to extract discriminative features
    from time series and feeds them into an MLP for classification.
    
    Args:
        input_size (int): Length of input time series
        num_classes (int): Number of output classes
        in_dim (int): Number of input channels/features
        d_model (int): Size of hidden layers
        num_shapelets (int): Total number of shapelets to learn
        shapelet_lengths (List[int]): List of different shapelet lengths to use
        shapelet_counts (Optional[List[int]]): Number of shapelets per length
        n_hidden_layers (int): Number of hidden layers in the classifier
        dropout_p (float): Dropout probability
        learning_rate (float): Learning rate for optimizer
        dataset_name (DATASET_NAMES): Name of the dataset
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        in_dim: int,
        d_model: int,
        num_shapelets: int = 100,
        shapelet_lengths: Optional[List[int]] = None,
        shapelet_counts: Optional[List[int]] = None,
        n_hidden_layers: int = 2,
        dropout_p: float = 0.2,
        learning_rate: float = 1e-3,
        hyperparams_search_str: str = "NoHyperparamSearch",
        dataset_name: DATASET_NAMES = "Welding",
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_classes,
            in_dim=in_dim,
            d_model=d_model,
            n_hidden_layers=n_hidden_layers,
            dropout_p=dropout_p,
            learning_rate=learning_rate,
            hyperparams_search_str=hyperparams_search_str,
            dataset_name=dataset_name,
        )
        
        # Default shapelet parameters if not provided
        if shapelet_lengths is None:
            # Create shapelets of different lengths
            shapelet_lengths = [max(3, input_size // 10), max(5, input_size // 5), max(7, input_size // 3)]
        
        if shapelet_counts is None:
            # Equal distribution of shapelets across different lengths
            num_lengths = len(shapelet_lengths)
            shapelet_counts = [num_shapelets // num_lengths] * num_lengths
            # Add any remainder to the last group
            shapelet_counts[-1] += num_shapelets - sum(shapelet_counts)
        
        # Shapelet layer to extract features
        self.shapelet_layer = ShapeletLayer(
            in_channels=in_dim,
            num_shapelets=num_shapelets,
            shapelet_lengths=shapelet_lengths,
            num_shapelet_per_length=shapelet_counts,
        )
        
        # MLP classifier
        layers = [nn.Linear(num_shapelets, d_model), nn.ReLU(), nn.Dropout(dropout_p)]
        
        for _ in range(n_hidden_layers - 1):
            layers.extend([
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ])
        
        layers.append(nn.Linear(d_model, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input time series [batch_size, in_dim, input_size]
            
        Returns:
            torch.Tensor: Classification logits [batch_size, num_classes]
        """
        # Get shapelet distances
        shapelet_features = self.shapelet_layer(x)
        
        # Apply classifier to get logits
        logits = self.classifier(shapelet_features)
        
        return logits
