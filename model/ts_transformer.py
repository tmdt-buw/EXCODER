import torch
import torch.nn as nn
import math
import logging
from torch.nn import functional as F
from model.embedding import DataEmbedding
from model.transformer_block import Block
from model.transformer_base import TransformerDecoderBase


class TS_Transformer(TransformerDecoderBase):
    def __init__(
        self,
        dataset_name: str,
        d_model: int = 64,
        seq_len: int = 100,
        in_dim: int = 1,
        n_classes: int = 2,
        n_blocks: int = 2,
        n_head: int = 6,
        res_dropout=0.1,
        att_dropout=0.0,
        learning_rate: float = 1e-3,
        dropout_p: float = 0.1,
        class_h_bias: bool = False,
        embed_type: str = "timeF",
        freq: str = "h",
        hyperparams_search_str: str = "NoHyperparamSearch",
        hash_model: str = "",
    ):
        self.hyperparams_search_str = hyperparams_search_str
        self.hash_model = hash_model
        self.input_size = seq_len
        self.in_dim = in_dim
        self.num_classes = n_classes
        super().__init__(
            dataset_name=dataset_name,
            d_model=d_model,
            embedding_classes=1,
            seq_len=seq_len,
            n_blocks=n_blocks,
            n_head=n_head,
            res_dropout=res_dropout,
            att_dropout=att_dropout,
            n_classes=n_classes,
            learning_rate=learning_rate,
            class_h_bias=class_h_bias,
        )

        self.task = "classification"

        self.embedding = DataEmbedding(
            c_in=in_dim,
            d_model=d_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout_p,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_classes)
        """
        x = self.embedding(x, None)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.class_head.linear_1(x)
        x = self.class_head.activation(x.squeeze(-1))
        logits = self.class_head.linear_2(x)
        return logits


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
        logits = self(x)
        loss = self.loss_cross_entropy(logits, y)
        return loss, logits, y


    @staticmethod
    def get_attention_hook(weights_list):
        def hook(module, input, output):
            # Get attention weights from the module
            # att shape: (batch_size, n_head, seq_len, seq_len)
            weights_list.append(module.last_attn_weights["normalized"])

        return hook

    @staticmethod
    def compute_attention_flow(attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention flow from attention weights.

        Args:
            attention_weights (torch.Tensor): Attention weights of shape (batch_size, n_layers, n_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: Attention flow tensor of shape (batch_size, seq_len)
        """
        # Add residual connections
        attention_weights = 0.5 * attention_weights + 0.5 * torch.eye(attention_weights.shape[-1])

        # Average over heads
        attention_weights = attention_weights.mean(dim=[2])  # (batch_size, n_layers, n_heads, seq_len, seq_len)

        # Initialize attention flow
        attention_flow = attention_weights[:, -1, :]  # (batch_size, seq_len)

        # Compute attention flow recursively
        for l in range(attention_weights.shape[1] - 2, -1, -1):
            attention_flow = attention_flow @ attention_weights[:, l, :, :]

        return attention_flow


    def get_attention_weights(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for each transformer block during forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Attention weight tensor for each input token (batch_size, seq_len)
                - Predicted sequence of shape (batch_size, sequence_length)
        """
        attention_weights = []

        # Register hooks for each attention block
        hooks = []
        for block in self.transformer.h:
            hooks.append(
                block.attn.register_forward_hook(self.get_attention_hook(attention_weights))
            )

        # Forward pass
        logits = self(x)  # run forward pass to trigger hooks

        pred = torch.argmax(logits, dim=-1)
        # Remove hooks
        for hook in hooks:
            hook.remove()
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_weights = attention_weights.mean(dim=[1, 2, 3])
        # repeat attention weights x_dim times
        attention_weights = attention_weights.repeat(1, x.shape[2])
        return attention_weights, pred


    def get_attention_flows(
        self, x: torch.Tensor, generate: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for each transformer block during forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            generate (bool): Whether to generate a sequence or perform classification

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Attention weight tensor for each input token (batch_size, seq_len)
                - Predicted sequence of shape (batch_size, sequence_length)
        """
        attention_weights = []

        # Register hooks for each attention block
        hooks = []
        for block in self.transformer.h:
            hooks.append(
                block.attn.register_forward_hook(self.get_attention_hook(attention_weights))
            )

        # Forward pass
        logits = self(x)  # run forward pass to trigger hooks

        pred = torch.argmax(logits, dim=-1)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        attention_weights = torch.stack(attention_weights, dim=1)

        # Modify here to compute attention flow
        attention_flow = self.compute_attention_flow(attention_weights.cpu())

         # Average over heads
        attention_flow = attention_flow.mean(dim=-1)
        attention_flow = attention_flow.repeat(1, x.shape[2])
        return attention_flow, pred