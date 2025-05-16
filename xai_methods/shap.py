# # import shap
import torch
import numpy as np
import pandas as pd  # Import pandas
from xai_methods.xai_blackbox_base import BlackBoxExplainer


class SHAP(BlackBoxExplainer):
    """
    A class to explain PyTorch models using the SHAP (SHapley Additive exPlanations) method, inheriting from BlackBoxExplainer.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be explained.
        explainer (shap.Explainer): The SHAP explainer object.
        background_data (torch.Tensor or np.ndarray): Background data for the explainer (optional, can be inferred).
        feature_names (list): List of feature names (optional).
    """

    def __init__(
        self,
        model,
        model_type: str,
        dataset_type: str,
        use_latent_input: bool,
        background_data: np.array,
        conf: dict[str, any] | None = None,
    ):
        """
        Initializes the ShapExplainer.

        Args:
            model (torch.nn.Module): The PyTorch model to be explained.
            model_type (str): Type of the model.
            dataset_type (str): Type of the dataset.
            use_latent_input (bool): Whether to use latent input.
            background_data (torch.Tensor or np.ndarray, optional):
                Background data used to integrate out features for SHAP value calculation.
            conf (dict, optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)
        raise NotImplementedError("SHAP is not implemented")
#         if self.model_type == "VQ-VAE_Transformer" or self.model_type == "MLP":
#             raise NotImplementedError(f"{self.model_type} is not supported for SHAP")

#         # Convert model to eval mode
#         self.model.eval()
        
#         # Create a masker from the background data
#         self.background_data = shap.utils.sample(background_data, nsamples=1_000)
#         self.background_data = self._to_tensor(self.background_data)
        
#         self.explainer = shap.DeepExplainer(self.model, self.background_data)

#         self.feature_names = None
#         self.input_size = self.model.input_size
#         self.input_dim = self.model.in_dim
#         self.num_class = self.model.num_classes

#     def predict_fn(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         Predicts class labels for the input tensor.

#         Args:
#             input_tensor: Input data tensor

#         Returns:
#             torch.Tensor: Predicted class labels
#         """
#         logits = self.model.forward(input_tensor)
#         pred = torch.argmax(logits, dim=-1)
#         return pred

#     def _to_tensor(self, data):
#         """
#         Converts data to a PyTorch tensor.

#         Args:
#             data (torch.Tensor, np.ndarray, or list): Data to convert.

#         Returns:
#             torch.Tensor: Data as a PyTorch tensor.
#         """
#         if isinstance(data, torch.Tensor):
#             return data
#         elif isinstance(data, np.ndarray):
#             return torch.from_numpy(data).float()
#         elif isinstance(data, list):
#             return torch.tensor(data).float()
#         else:
#             raise TypeError("Data must be a torch.Tensor, np.ndarray, or list.")

    def explain(
        self,
        data,
        save_to_pickle=False,
        save_path: str = "shap_explanations.pkl",
        **kwargs,
    ):
        """
        Calculates SHAP values for the given data. Overrides the abstract explain method of BlackBoxExplainer.

        Args:
            data (torch.Tensor, np.ndarray, or list): Data for which to calculate SHAP values.
            save_to_pickle (bool): Whether to save the explanations to a pickle file.
            save_path (str): Path to save the explanations to.
            **kwargs: Additional keyword arguments to pass to the explainer's `shap_values` method.

        Returns:
            np.ndarray: SHAP values for the predicted class, shape (n_samples, timesteps, features)
        """
        raise NotImplementedError("SHAP is not implemented")
#         # Get SHAP values for all classes
#         output = self.explainer(data, **kwargs).values  # Shape: (200, 200, 1, 5)
        
#         # Get predictions
#         pred = self.predict_fn(data).cpu().numpy()  # Shape: (200,)
        
#         # Select SHAP values for predicted classes using advanced indexing
#         batch_indices = np.arange(len(pred))
#         output = output[batch_indices, :, :, pred]  # Shape: (200, 200, 1)

#         if save_to_pickle:
#             output = output.reshape(-1, self.input_size, self.input_dim)
#             self.save(output, save_path)
#         return output

    