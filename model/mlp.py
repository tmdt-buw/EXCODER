import torch
from torch import nn
import torch.nn.functional as F
from model.classification_base import ClassificationLightningModule
from params import DATASET_NAMES


class MLP(ClassificationLightningModule):
    """
    Multi-Layer Perceptron model for classification tasks.

    A simple feedforward neural network with configurable hidden layers, normalization,
    and dropout. Inherits from ClassificationLightningModule for training functionality.

    Args:
        input_size (int): Size of input features
        in_dim (int): Input dimension
        num_class (int): Number of output classes
        learning_rate (float): Learning rate for optimization
        n_hidden_layers (int): Number of hidden layers
        d_model (int): Dimension of hidden layers
        dropout_p (float): Dropout probability
        use_layer_norm (bool): Whether to use layer normalization
        use_latent_input (bool): Whether to use latent input from a latent space (VQ-VAE or DVAE)
        num_latent_tokens (int): Number of latent tokens
        hyperparams_search_str (str): String identifier for hyperparameter search
        dataset_name (DATASET_NAMES): Name of the dataset
        hash_model (str): Hash identifier for the model
    """

    def __init__(
        self,
        input_size: int,
        in_dim: int,
        num_class: int,
        learning_rate: float = 1e-3,
        n_hidden_layers: int = 2,
        d_model: int = 128,
        dropout_p: float = 0.1,
        use_layer_norm: bool = False,
        use_latent_input: bool = False,
        num_latent_tokens: int = 100,
        hyperparams_search_str: str = "NoHyperparamSearch",
        dataset_name: DATASET_NAMES = "Welding",
        hash_model: str = "",
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_class,
            in_dim=in_dim,
            d_model=d_model,
            n_hidden_layers=n_hidden_layers,
            dropout_p=dropout_p,
            learning_rate=learning_rate,
            hyperparams_search_str=hyperparams_search_str,
            dataset_name=dataset_name,
            hash_model=hash_model,
        )

        self.num_latent_tokens = num_latent_tokens
        self.use_latent_input = use_latent_input
        self.embedding = (
            nn.Embedding(num_latent_tokens, d_model)
            if use_latent_input
            else nn.Identity()
        )

        self.layer_list = []
        self.layer_list.extend(
            [
                nn.Linear(
                    (
                        input_size * d_model
                        if use_latent_input
                        else input_size * in_dim
                    ),
                    d_model,
                ),
                nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
                nn.GELU(),
            ]
        )

        for _ in range(n_hidden_layers - 1):
            self.layer_list.append(nn.Linear(d_model, d_model))
            self.layer_list.append(
                nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
            )
            self.layer_list.append(nn.GELU())

        self.layer_list.extend(
            [
                nn.Dropout(dropout_p),
                nn.Linear(d_model, num_class),
            ]
        )

        self.layers = nn.Sequential(*self.layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size * in_dim)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = x.reshape(x.shape[0], -1)
        if self.use_latent_input:
            x = self.embedding(x).reshape(x.shape[0], -1)
        x = self.layers(x)
        return x

    def forward_get_saliency(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that computes saliency maps for input attribution.

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target labels

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Model predictions
                - Saliency scores for input features
        """
        x = x.reshape(x.shape[0], -1)
        if self.use_latent_input:
            embedding = self.embedding(x)
        else:
            embedding = x
        embedding.retain_grad()
        logits = self.layers(embedding.reshape(x.shape[0], -1))
        loss = self.loss(logits, y)
        loss.backward()

        saliency_embed = embedding.grad.abs().sum(dim=-1)
        prediction = F.log_softmax(logits, dim=1).argmax(dim=1)
        return prediction, saliency_embed
