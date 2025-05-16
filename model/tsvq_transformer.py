import math
from typing import Literal
import logging
import torch
from torch import nn
import torch.nn.functional as F
from model.embedding import PositionalEmbedding
from model.transformer_block import Block
from model.transformer_base import TransformerDecoderBase
from model.vq_vae_patch_embed import (
    PatchEmbedding,
    PatchEmbeddingInverse,
    CNNBlock,
    SepCNNBlock,
)
from model.vector_quantizer import ResidualVQLightning


class VQEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_resblocks: int = 3,
        dropout_p: float = 0.1,
        batch_norm: bool = False,
    ):
        super(VQEncoder, self).__init__()
        # Single convolutional layer blocks whose weights will be shared
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=hidden_dim)

        self.encoder = nn.Sequential(
            CNNBlock(
                embed_dim=hidden_dim,
                n_resblocks=n_resblocks,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
            SepCNNBlock(
                hidden_dim=hidden_dim, embedding_dim=embed_dim
            ),  # num_embeddings is the embedding dimension for dvae
        )

    def forward(self, x):
        x = self.patch_embed(x)
        z_e = self.encoder(x)
        return z_e


class VQDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        hidden_dim: int,
        input_dim: int,
        n_resblocks: int = 3,
        dropout_p: float = 0.1,
        batch_norm: bool = False,
    ):
        super(VQDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            CNNBlock(
                embed_dim=hidden_dim,
                seperate=False,
                n_resblocks=n_resblocks,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
        )

        self.reverse_patch_embed = PatchEmbeddingInverse(
            patch_size=patch_size, embed_dim=hidden_dim, input_dim=input_dim
        )

    def forward(self, x):
        x = self.decoder(x.permute(0, 2, 1))
        x = self.reverse_patch_embed(x)
        return x


class TSVQTransformer(TransformerDecoderBase):
    def __init__(
        self,
        dataset_name: str,
        d_model: int = 64,
        seq_len: int = 100,
        data_dim: int = 2,
        embedding_classes: int = 131,
        n_blocks: int = 2,
        n_head: int = 6,
        res_dropout=0.1,
        att_dropout=0.0,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        class_h_bias: bool = False,
        patch_size: int = 25,
        vq_hidden_dim: int = 2,
        vq_n_resblocks: int = 3,
        beta: float = 0.05,
        theta: float = 1,
        gamma: float = 1,
    ):
        super().__init__(
            dataset_name=dataset_name,
            d_model=d_model,
            embedding_classes=embedding_classes,
            seq_len=seq_len,
            n_blocks=n_blocks,
            n_head=n_head,
            res_dropout=res_dropout,
            att_dropout=att_dropout,
            n_classes=n_classes,
            learning_rate=learning_rate,
            class_h_bias=class_h_bias,
        )
        self.task: Literal["classification", "generate"] = "classification"
        self.theta = theta
        self.theta_decay = 0.05
        self.gamma = gamma
        self.gamma_decay = 0.05
        self.beta = beta
        self.start_token = 0
        self.unknown_token = 1
        self.end_token = 2
        self.data_dim = data_dim
        self.vq_n_resblocks = vq_n_resblocks
        self.n_special_tokens = 2

        self.token_embeddings = nn.Embedding(self.n_special_tokens, d_model)
        self.pos_embeddings = PositionalEmbedding(d_model=d_model, max_len=seq_len)

        self.vq_encoder = VQEncoder(
            patch_size=patch_size,
            embed_dim=d_model,
            hidden_dim=vq_hidden_dim,
            n_resblocks=vq_n_resblocks,
        )

        self.vq = ResidualVQLightning(
            num_quantizers=1,
            e_dim=d_model,
            n_e=embedding_classes,
            kmeans_init=True,
            kmeans_iters=20,
            threshold_ema_dead_code=2,
        )

        self.vq_decoder = VQDecoder(
            patch_size=patch_size,
            embed_dim=d_model,
            hidden_dim=vq_hidden_dim,
            input_dim=data_dim,
            n_resblocks=vq_n_resblocks,
        )

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(res_dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model=d_model,
                            seq_len=seq_len,
                            n_head=n_head,
                            res_dropout=res_dropout,
                            att_dropout=att_dropout,
                        )
                        for _ in range(n_blocks)
                    ]
                ),
                ln_f=nn.LayerNorm(d_model),
            )
        )
        self.lm_head = nn.Linear(d_model, embedding_classes + 3, bias=False)

        class_head_module_dict = dict(
            linear_1=nn.Linear(d_model, 1, bias=class_h_bias),
            activation=nn.GELU(),
            linear_2=nn.Linear(seq_len, n_classes, bias=class_h_bias),
        )
        self.class_head = nn.ModuleDict(class_head_module_dict)
        # initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))
        self.log_num_params()

    def log_num_params(self) -> None:
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logging.info(
            f"Transformer Blocks number of parameters: {(n_params / 1e6):.4f}M"
        )

    def create_input_embedding(
        self, x: torch.Tensor, prob_missing: float = 0.0
    ) -> torch.Tensor:
        start_vec = self.token_embeddings(
            torch.tensor([self.start_token], device=self.device)
        )
        unknown_vec = self.token_embeddings(
            torch.tensor([self.unknown_token], device=self.device)
        )
        start_vec = start_vec.repeat(x.shape[0], 1, 1)
        # print(f"start_vec: {start_vec.shape} - unknown_vec: {unknown_vec.shape}")
        # TODO: add prob_missing as matrix in a vectroized manner
        x_input = torch.cat([start_vec, x], dim=1)
        embed = self.pos_embeddings(x_input)
        return x_input + embed

    def create_output_embedding(self, x_index: torch.Tensor) -> torch.Tensor:
        # we need to add the num_embeddings from the quantizer to the end token because the token index is already taken by the quantizer
        end_vec = torch.tensor([self.end_token]) + self.embedding_classes
        end_vec = end_vec.repeat(x_index.shape[0], 1).to(self.device)
        x_output = torch.cat([x_index.squeeze(-1), end_vec], dim=1)
        return x_output

    def forward(
        self, x: torch.Tensor, classification: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            classification (bool): Whether to perform classification

        Returns:
            ts_pred (torch.Tensor): Time series prediction of shape (batch_size, sequence_length, input_size*input_dim)
            class_logits (torch.Tensor): Class logits of shape (batch_size, n_classes)
            seq_loss (torch.Tensor): Sequence loss
            enc_embedding_loss (torch.Tensor): Encoder embedding loss
        """
        z_e = self.vq_encoder(x)
        enc_embedding_loss, z_q, _, _, indices = self.vq(z_e)
        ts_pred = self.vq_decoder(z_q)
        ts_loss = self.loss_mse(ts_pred, x)
        
        x_output = self.create_output_embedding(indices)
        
        z_q = self.create_input_embedding(z_q)
        for block in self.transformer.h:
            z_q = block(z_q)
        z_q = self.transformer.ln_f(z_q)
        seq_logits = self.lm_head(z_q)
        seq_loss = self.loss_cross_entropy(seq_logits, x_output, ignore_index=-1)

        # set prob num_embeddings to zero and cut last token
        # seq_logits[:, :, self.embedding_classes :] = 0
        # seq_logits = seq_logits[:, :-1, :]

        # use vq_decoder to get the predicted time series
        # x_id = F.log_softmax(seq_logits, dim=-1)
        # x_id = x_id.argmax(dim=-1)
        # pred_z_q = self.vq.get_codebook_vector(x_id)
        # ts_pred = self.vq_decoder(pred_z_q)

        if classification:
            z_q = self.class_head.linear_1(z_q)
            z_q = self.class_head.activation(z_q.squeeze(-1))
            class_logits = self.class_head.linear_2(z_q)
            return ts_loss, class_logits, seq_loss, enc_embedding_loss
        else:
            return ts_loss, None, seq_loss, enc_embedding_loss

    def get_gamma(self) -> float:
        """
        Get the gamma value for the current epoch.
        -> gets smaller over time
        """
        return self.gamma * (1 - self.gamma_decay) ** self.current_epoch

    def get_theta(self) -> float:
        """
        Get the theta value for the current epoch.
        -> gets larger over time
        """
        return 1 - self.theta * (1 - self.theta_decay) ** self.current_epoch

    def step_task_gen(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute the loss for the generation task.

        Args:
            batch (tuple): A tuple containing the input data and labels.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        x = batch
        ts_loss, _, seq_loss, enc_embedding_loss = self(x, classification=False)

        self.log("mse_loss", ts_loss.item())
        self.log("enc_embedding_loss", enc_embedding_loss.item())
        self.log("seq_loss", seq_loss.item())

        loss = self.beta * enc_embedding_loss + (1 - self.beta) * (
            self.theta * seq_loss + self.gamma * ts_loss
        )
        return loss, _, _

    def step_task_class(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute the loss for the classification task.

        Args:
            batch (tuple): A tuple containing the input data and labels.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        x, y = batch
        ts_loss, class_logits, seq_loss, enc_embedding_loss = self(
            x, classification=True
        )

        class_loss = self.loss_cross_entropy(class_logits, y)

        self.log("mse_loss", ts_loss.item())
        self.log("enc_embedding_loss", enc_embedding_loss.item())
        self.log("seq_loss", seq_loss.item())
        self.log("class_loss", class_loss.item())

        loss = self.beta * enc_embedding_loss + (1 - self.beta) * (
            self.theta * seq_loss + self.gamma * ts_loss + 0.8 * class_loss
        )
        return loss, class_logits, y
