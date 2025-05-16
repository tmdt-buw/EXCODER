import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.nn import functional as F
from xai_methods.xai_blackbox_base import BlackBoxExplainer
import math
from model.transformer_block import Block, CausalSelfAttention
from model.transformer_decoder import MyTransformerDecoder
from model.TimesNet import TimesNet
from model.DLinear import DLinear
from model.mlp import MLP


class DeepLift(BlackBoxExplainer):
    """
    Implementation of DeepLIFT (Deep Learning Important FeaTures) algorithm for attribution.
    Computes contribution scores for each input feature by comparing activations
    to a reference baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        dataset_type: str,
        use_latent_input: bool,
        baseline: torch.Tensor,
        conf: dict[str, any] | None = None,
    ):
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)
        self.baseline = baseline
        self.use_latent_input = use_latent_input
        self.reference_output = self.model(self.baseline)
        self.model_type = model_type
        self.supported_models = []

        if self.model_type not in self.supported_models:
            raise ValueError(f"Model type {self.model_type} not supported")

    def attribute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute DeepLift attributions for the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor to explain

        Returns:
            torch.Tensor: Attribution scores for the input
        """
        pass

    def compute_contributions_based_on_layer_type(
        self, input_tensor: torch.Tensor, layer: nn.Module
    ) -> torch.Tensor:
        """
        Compute DeepLift contribution values based on the layer type for MLP and Transformer models.

        Args:
            input_tensor (torch.Tensor): Input tensor
            layer (nn.Module): Layer from the model

        Returns:
            torch.Tensor: DeepLift contribution values for the layer
        """
        pass

    def explain(
        self,
        input_tensor: torch.Tensor,
        save_to_pickle: bool = False,
        save_path: str = "DeepLIFT_explanations.pkl",
    ) -> torch.Tensor:
        """
        Explain the model's prediction for the given input tensor using DeepLIFT.

        Args:
            input_tensor: Input data tensor (Batch, Input_Size, Input_Dim)

        Returns:
            torch.Tensor: Explanation tensor (Batch, Input_Size, Input_Dim)
        """
        # Generate baseline/reference
        if self.baseline is None:
            self.baseline = torch.zeros_like(input_tensor)

        # Compute attributions using DeepLIFT
        pass        
        # if save_to_pickle:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(attributions, f)
        # return attributions
