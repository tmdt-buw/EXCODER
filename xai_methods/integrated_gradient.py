import torch
import torch.nn as nn
from typing import Optional
from xai_methods.xai_blackbox_base import BlackBoxExplainer
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


class IntegratedGradients(BlackBoxExplainer):
    """Implements Integrated Gradients method for attribution-based explanations.

    Computes attributions by integrating gradients along a straight path from a baseline
    to the input, supporting various model architectures including MLPs, Transformers and VAEs.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        dataset_type: str,
        use_latent_input: bool,
        steps: int = 50,
        batch_size: int = 50,
        baseline: Optional[torch.Tensor] = None,
        conf: dict[str, any] | None = None,
    ):
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)
        self.steps = steps
        self.baseline = baseline
        self.batch_size = batch_size
        self.input_size = baseline.shape[1]
        self.input_dim = baseline.shape[2] if len(baseline.shape) > 2 else 1

    def _get_gradients(
        self, inputs: torch.Tensor, target_class: torch.Tensor
    ) -> torch.Tensor:
        """Computes gradients of model output with respect to input embeddings.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, features)
            target_class: Target class tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Computed gradients with respect to input embeddings
                Shape matches input shape: (batch_size, seq_len) or (batch_size, seq_len, features)

        Raises:
            ValueError: If model_type is not supported
        """
        loss_func = nn.CrossEntropyLoss()

        if self.model_type == "MLP":
            inputs.requires_grad = True
            embed = self.model.embedding(inputs)
            out = self.model.layers(embed.reshape(len(inputs), -1))
            loss_func(out, target_class).backward()
            gradients = inputs.grad

        elif self.model_type in ["VQ-VAE_MLP", "DVAE_MLP", "SAX_MLP"]:
            embed = self.model.embedding(inputs)
            embed.retain_grad()
            out = self.model.layers(embed.reshape(len(inputs), -1))
            embed.retain_grad()
            loss_func(out, target_class).backward()
            gradients = embed.grad.sum(dim=-1).unsqueeze(-1)

        elif self.model_type == "DLinear":
            inputs.requires_grad = True
            out = self.model(inputs)
            inputs.retain_grad()
            loss_func(out, target_class).backward()
            gradients = inputs.grad

        elif self.model_type == "TimesNet":
            inputs.requires_grad = True
            out = self.model(inputs)
            inputs.retain_grad()
            loss_func(out, target_class).backward()
            gradients = inputs.grad

        elif self.model_type in ["VQ-VAE_Transformer", "DVAE_Transformer"]:
            x = self.model.embedding(inputs)
            orig_x = x
            orig_x.retain_grad()
            # Forward pass through transformer
            for block in self.model.transformer.h:
                x = block(x)
            x = self.model.transformer.ln_f(x)
            x = self.model.class_head.linear_1(x)
            x = self.model.class_head.activation(x.squeeze(-1))
            x = self.model.class_head.linear_2(x)

            loss_func(x, target_class).backward()
            gradients = orig_x.grad.sum(dim=-1).unsqueeze(-1)
        elif self.model_type == "TS_Transformer":
            inputs.requires_grad = True
            out = self.model(inputs)
            loss_func(out, target_class).backward()
            gradients = inputs.grad
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        return gradients

    def get_order_tokens(self) -> torch.Tensor:
        """Gets tokens ordered by their distance to the unknown token.

        Returns:
            Tensor of token indices sorted by distance to unknown token
        """

        if self.model_type == "VQ-VAE_MLP" or self.model_type == "DVAE_MLP" or self.model_type == "SAX_MLP":
            embedding_space = self.model.embedding.weight
            idx_unknown = self.model.num_latent_tokens - 1
        elif (
            self.model_type == "DVAE_Transformer"
            or self.model_type == "VQ-VAE_Transformer"
        ):
            embedding_space = self.model.embedding.latent_embedding.weight
            idx_unknown = self.model.embedding_classes - 1
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        unknown_token = embedding_space[idx_unknown]

        # get distance to unknown token
        distances = torch.norm(embedding_space - unknown_token, dim=-1)
        # order tokens by distance to unknown token
        order_tokens = torch.argsort(distances)
        return order_tokens.cpu()

    def get_interpolated_latent_input(
        self, alpha: float, input_x: torch.Tensor, order_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Interpolates between tokens based on their distance to the unknown token.

        Args:
            alpha: Interpolation factor between 0 and 1
            input_x: Input tensor containing token indices
            order_tokens: Tensor of tokens ordered by distance to unknown token

        Returns:
            Tensor of interpolated token indices
        """
        input_len = input_x.shape[0]

        # get indices of input_x in order_tokens
        order_tokens_expanded = order_tokens.unsqueeze(
            0
        )  # Shape: (1, num_order_tokens)
        input_x_expanded = input_x.unsqueeze(2)  # Shape: (1, num_input_x, 1)
        # Compare element-wise
        equal_mask = (
            input_x_expanded == order_tokens_expanded
        )  # Shape: (1, num_input_x, num_order_tokens)

        # Find indices where the mask is True
        #    Since nonzero() returns indices in a flattened way across multiple dimensions,
        #    we only care about the last dimension which represents the index within order_tokens.
        indices = torch.nonzero(equal_mask)

        # Extract the indices corresponding to order_tokens from the last dimension
        result_indices = indices[:, -1]
        result_indices = result_indices.reshape(input_len, -1)

        # result_indices * alpha to int
        result_indices = (result_indices * alpha).int()
        return order_tokens[result_indices]

    def explain(
        self,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        save_to_pickle: bool = False,
        save_path: str = None,
    ) -> torch.Tensor:
        """
        Generate integrated gradients explanation for the model's prediction.

        Args:
            dataset: Input data tensor
            target: Target class tensor for explanation


        Returns:
            torch.Tensor: Attribution scores
        """

        # Move input to same device as model

        # If no baseline provided, use zero tensor
        if self.baseline is None:
            self.baseline = torch.zeros_like(input_tensor)

        # Generate steps interpolated inputs between baseline and input
        alphas = torch.linspace(0, 1, self.steps)
        if self.use_latent_input:
            order_tokens = self.get_order_tokens()

            interpolated = torch.stack(
                [
                    self.get_interpolated_latent_input(
                        alpha, input_tensor, order_tokens
                    )
                    for alpha in alphas
                ]
            )
        else:
            interpolated = torch.stack(
                [
                    self.baseline + alpha * (input_tensor - self.baseline)
                    for alpha in alphas
                ]
            )

        # Reshape to (steps * ds_size, seq_len, features)
        ds_size = input_tensor.size(0)
        interpolated = interpolated.view(-1, *input_tensor.shape[1:])

        targets = target.repeat_interleave(self.steps)

        # Create dataset and dataloader for interpolated inputs
        dataset = TensorDataset(interpolated, targets)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,  
            num_workers=4  
        )

        # Process all models in batches
        all_gradients = torch.empty(size=(len(interpolated), self.input_size, self.input_dim))
        
        for i, (batch, target_batch) in enumerate(tqdm(dataloader, total=len(dataloader))):
            batch = batch.to(self.model.device)
            target_batch = target_batch.to(self.model.device)
            grad_batch = self._get_gradients(batch, target_batch)
            start_idx = i * self.batch_size
            end_idx = start_idx + len(batch)  # handles last batch which might be smaller
            all_gradients[start_idx:end_idx] = grad_batch.cpu()

        # Reshape gradients back to (steps, batch_size, seq_len, features)
        gradients = all_gradients.view(self.steps, ds_size, *input_tensor.shape[1:])

        # Calculate integrated gradients using trapezoidal rule
        integrated_gradients = (
            gradients[:-1] + gradients[1:]
        ) / 2.0  # Average consecutive pairs

        attributions = torch.mean(integrated_gradients, dim=0)

        attributions = attributions.abs()

        if save_to_pickle:
            attributions = attributions.reshape(-1, self.input_size, self.input_dim)
            self.save(attributions, save_path)
        return attributions
