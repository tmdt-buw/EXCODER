from abc import abstractmethod
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



class BlackBoxExplainer:
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        dataset_type: str,
        use_latent_input: bool,
        conf: dict[str, any] | None = None,
    ):
        self.model = model
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.use_latent_input = use_latent_input
        self.conf = conf

    @abstractmethod
    def explain(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Explain the model's prediction for the given input tensor.

        Args:
            input_tensor: Input data tensor

        Returns:
            torch.Tensor: Tuple of (Model Output, Explanation)

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError

    def save(self, explanations: torch.Tensor | np.ndarray, path: str | Path) -> None:
        """
        Save the explanations to a pickle file, together with the explainer's parameters.

        Args:
            explanations: Explanations in torch.Tensor or numpy.ndarray format
            path: Path where to save the pickle file

        Returns:
            None
        """
        if isinstance(explanations, torch.Tensor):
            explanations = explanations.cpu().numpy()

        # Base dictionary with common fields
        dict_to_save = {
            "explanations": explanations,
            "use_latent_input": self.use_latent_input,
            "model_type": self.model_type,
            "dataset_type": self.dataset_type,
            **({"conf": self.conf} if self.conf is not None else {})
        }

        with open(path, "wb") as f:
            pickle.dump(dict_to_save, f)

    @torch.no_grad()
    def classify(self, x: torch.Tensor, return_numpy: bool = True) -> np.ndarray:
        """
        Predict probabilities using the model.

        Args:
            x: Input tensor in Shape (Batch, Input_Size, Input_Dim)

        Returns:
            np.ndarray: Predicted Probabilities in Shape (Batch, Num_Classes)
        """
        x = x.to(self.model.device)
        if self.model_type in ["DVAE_Transformer", "VQ-VAE_Transformer"]:
            logits = self.model(x.squeeze(-1), generate=False)
        else:
            logits = self.model(x)
        return F.softmax(logits, dim=1).cpu().numpy() if return_numpy else F.softmax(logits, dim=1).cpu()

    def print_model(self) -> None:
        """Print model hyperparameters."""
        print(self.model.hparams)
