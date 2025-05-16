import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from tqdm import tqdm
from xai_methods.xai_blackbox_base import BlackBoxExplainer


def generate_rise_masks(
    smooth_edges: bool,
    seq_len: int,
    num_masks: int,
    min_masking_value: float = 0.85,
    n_masked_percentage: float = 0.1,
    plot: bool = False,
):
    """
    Generates RISE masks for input images.
    Args:
        smooth_edges (int): Whether to smooth the edges of the masks
        seq_len (int): Length of the high-resolution mask
        num_masks (int): Number of masks to generate. Default is 2.
        min_masking_value (float): Minimum masking value for the masked pixels. Default is 0.85. 0 equals maximum masking.
        n_masked_percentage (int): Number of pixels to mask. If provided, the specified percentage of pixels will be masked randomly. Default is 10 percent.
        plot (bool): Whether to plot the masks. Default is True.
    Returns:
        torch.Tensor: Generated masks.
    """
    # allow masks with missing category
    if smooth_edges:
        low_res_mask_length = round(seq_len / 2)
        masks = torch.ones((num_masks, low_res_mask_length))
        amount_to_mask = round(low_res_mask_length * n_masked_percentage)
        for i in range(num_masks):
            zero_indices = torch.randperm(low_res_mask_length)[:amount_to_mask]
            masks[i, zero_indices] = min_masking_value
        if plot:
            fig, ax = plt.subplots()
            fig.set_size_inches(2, 10)
            ax.set_title("Low Resolution Masks generated")
            ax.imshow(masks, cmap="Wistia")
            ax.set_yticks([])
            plt.show()
        masks = (
            F.interpolate(
                masks.unsqueeze(0).unsqueeze(0).float(),
                size=(num_masks, seq_len),
                mode="bilinear",
            )
            .squeeze(0)
            .squeeze(0)
        )
    else:
        amount_to_mask = round(seq_len * n_masked_percentage)
        masks = torch.ones((num_masks, seq_len), dtype=torch.float32)
        for i in range(num_masks):
            zero_indices = torch.randperm(seq_len)[:amount_to_mask]
            masks[i, zero_indices] = min_masking_value
    if plot:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 10)
        ax.set_title("Interpolated High Resolution Masks")
        ax.imshow(masks, cmap="Wistia")
        ax.set_yticks([])
        plt.show()
    return masks


def generate_distance_matrix(codebook):
    """
    Generates a distance matrix for the codebook.
    Args:
        codebook (torch.Tensor): Codebook tensor.
    Returns:
        torch.Tensor: Distance matrix.
    """
    distances = torch.cdist(codebook, codebook, p=2)
    sorted_distances = torch.argsort(distances, dim=1)
    return sorted_distances


def apply_masks(
    input,
    masks,
    distance_matrix,
    has_beginning_token=False,
    mask_with_missing_category=None,
    use_latent_input=False,
):
    """
    Applies the masks to the input.
    Args:
        input (torch.Tensor): Input tensor.
        masks (torch.Tensor): Masks to apply.
        distance_matrix (torch.Tensor): Distance matrix.
        has_beginning_token (bool): Whether the input has a beginning token. Default is True.
    Returns:
        torch.Tensor: Masked input.
    """
    # ignore beginning token of input
    if has_beginning_token:
        beginning_token = input[0]
        input = input[1:]

    if mask_with_missing_category:
        # check if masks only contain 0 or 1 (hard edges)
        assert torch.unique(masks).tolist() == [
            0,
            1,
        ], "Masks must only contain 0 or 1 when masking with missing category (set n_hard_eges to generate_rise_masks)."
        # apply mask and replace input with missing category whenever mask is 0, return dtype int
        input = input * masks.int().to(input.device) + mask_with_missing_category * (1 - masks.int().to(input.device))

    elif use_latent_input:
        num_vectors = distance_matrix.shape[1]
        scaled_indices = masks * (num_vectors - 1)
        position = torch.floor(scaled_indices).int() + 1
        input = distance_matrix[input, -position]

    else:
        input = input * masks.to(input.device)

    # add beginning token back to every mask
    if has_beginning_token:
        beginning_token = torch.tensor([beginning_token]).expand(masks.shape[0], -1)
        input = torch.cat((beginning_token, input), dim=1)
    return input


class Rise(BlackBoxExplainer):
    def __init__(
        self,
        model,
        min_masking_value=0,
        n_masked_percentage: float = 0.1,
        num_masks_per_instance: int = 1000,
        mask_with_missing_category: bool = True,
        missing_category: int = 32,
        classification_batch_size: int = 128,
        smooth_edges: bool = False,
        sorted_codebook_distances=None,
        use_latent_input: bool = True,
        model_type: str = "MLP",
        dataset_type: str = "CNC",
        conf=None,
    ):
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)

        self.input_size = self.model.input_size

        self.input_dim = self.model.in_dim if model_type not in ["VQ-VAE_MLP", "DVAE_MLP"] else 1
        self.min_masking_value = min_masking_value
        self.n_masked_percentage = n_masked_percentage
        self.num_masks_per_instance = num_masks_per_instance
        self.missing_category = missing_category
        self.classification_batch_size = classification_batch_size
        self.smooth_edges = smooth_edges
        self.sorted_codebook_distances = sorted_codebook_distances
        self.num_class = self.model.num_classes
        self.mask_with_missing_category = mask_with_missing_category

        if (self.mask_with_missing_category and self.min_masking_value != 0) or (
            self.smooth_edges and self.mask_with_missing_category
        ):
            raise ValueError(
                "If masking with missing category, min_masking_value must be 0 and smooth_edges must be False."
            )

    @torch.no_grad()
    def rise_classify_masked_inputs(
        self,
        input_tensor: torch.Tensor,
        used_masks: torch.Tensor,
        model_prediction: torch.Tensor,
        classification_batch_size: int = 128,
    ) -> torch.Tensor:
        probs = torch.zeros(len(input_tensor), self.num_class)
        for i in range(0, len(input_tensor), classification_batch_size):
            batch = input_tensor[i : i + classification_batch_size]
            pred = self.classify(batch, return_numpy=False)
            probs[i : i + classification_batch_size] = pred
        rise_map = torch.matmul(probs[:, model_prediction], used_masks)
        rise_map = rise_map / rise_map.sum()
        return rise_map

    def explain(
        self,
        input_tensor: torch.Tensor,
        save_to_pickle=False,
        save_path="rise_explanations.pkl",
    ) -> torch.Tensor:

        output = torch.empty(size=(len(input_tensor), self.input_size * self.input_dim))
        output = output.to(self.model.device)
        for i, instance in tqdm(enumerate(input_tensor), desc="Explaining instances", total=len(input_tensor)):
            masks_for_instance = generate_rise_masks(
                smooth_edges=self.smooth_edges,
                seq_len=self.input_size * self.input_dim,
                min_masking_value=self.min_masking_value,
                n_masked_percentage=self.n_masked_percentage,
                num_masks=self.num_masks_per_instance,
                plot=False,
            )
            masked_input = apply_masks(
                input=instance.reshape(1, self.input_size * self.input_dim),
                masks=masks_for_instance,
                distance_matrix=self.sorted_codebook_distances,
                mask_with_missing_category=self.missing_category,
            )
            pred_class = torch.argmax(self.classify(instance.unsqueeze(0), return_numpy=False))
            rise_maps = self.rise_classify_masked_inputs(
                input_tensor=masked_input.reshape(-1, self.input_size, self.input_dim),
                used_masks=masks_for_instance,
                model_prediction=pred_class,
                classification_batch_size=self.classification_batch_size,
            )
            output[i] = rise_maps
        output = output.reshape(len(input_tensor), self.input_size, self.input_dim)

        if save_to_pickle:
            self.save(output, save_path)

        return output
