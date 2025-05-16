import lightning.pytorch as pl
from vector_quantize_pytorch import ResidualVQ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical


class ResidualVQLightning(pl.LightningModule):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 0,
        threshold_ema_dead_code: int = 2,
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.num_quantizers = num_quantizers

        self.vq = ResidualVQ(
            num_quantizers=num_quantizers,
            dim=e_dim,
            codebook_size=n_e,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass of the VQ

        Args:
            x: (B, seq_len, embed_dim) input tensor

        Returns:
            z_q: (B, seq_len, embed_dim) quantized output tensor
            loss: (1) scalar tensor
            indices (B, seq_len) indices of z_q
        """
        z_q, indices, commit_loss = self.vq(x)
        # print(f"z_q: {z_q.shape}")
        # return loss, z_q, perplexity, min_encodings, min_encoding_indices
        return commit_loss, z_q, None, None, indices

    def get_codebook_vector(self, index_tensor: torch.Tensor):
        return self.vq.layers[0].codebook[index_tensor]


class DVAE_Discretizer(pl.LightningModule):
    def __init__(
        self,
        temperature:float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.save_hyperparameters()

    def dvae_discretize(self, x, temperature:float):
        z_logits = F.softmax(x, dim=-1)
        if self.training:
            z_q = RelaxedOneHotCategorical(temperature, z_logits).rsample()
        else:
            z_q = OneHotCategorical(z_logits).sample()
        return z_q 

    def forward(self, x):
        """
        Forward pass of the Vector Quantizer (VQ).

        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, embed_dim).
            temperature (float): Temperature parameter for the discretization process.

        Returns:
            torch.Tensor: Quantized output tensor of shape (B, seq_len, embed_dim).
            torch.Tensor: Scalar tensor representing the embedding loss for compatibility with vq_vae_patch_embed.py.
        """
        z_q = self.dvae_discretize(x, temperature=self.temperature)
        embedding_loss = torch.tensor(0.0)

        return embedding_loss, z_q, None, None, None