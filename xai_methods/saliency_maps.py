import torch.nn as nn
import torch
from abc import abstractmethod
from torch.nn import functional as F
import lightning.pytorch as pl
from xai_methods.xai_blackbox_base import BlackBoxExplainer
import numpy as np
from tqdm import tqdm


class Saliency_Maps(BlackBoxExplainer):
    def __init__(
        self,
        model,
        use_latent_input: bool = False,
        model_type: str = "MLP",
        dataset_type: str = "CNC",
        conf=None,
    ):
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)

    def explain(
        self,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        save_to_pickle=False,
        save_path="sm_explanations.pkl",
    ) -> torch.Tensor:
        loss_func = nn.CrossEntropyLoss()
        if self.model_type == "MLP":
            input_tensor.requires_grad = True
            input_tensor = input_tensor.to(self.model.device)
            target = target.to(self.model.device)
            embed = self.model.embedding(input_tensor)
            out = self.model.layers(embed.reshape(len(input_tensor), -1))
            input_tensor.retain_grad()
            loss_func(out, target).backward()
            output = input_tensor.grad.abs()
        elif self.model_type in ["VQ-VAE_MLP", "DVAE_MLP", "SAX_MLP"]:
            input_tensor = input_tensor.to(self.model.device)
            target = target.to(self.model.device)
            embed = self.model.embedding(input_tensor)
            embed.retain_grad()
            out = self.model.layers(embed.reshape(len(input_tensor), -1))
            loss_func(out, target).backward()
            output = embed.grad.abs().sum(dim=-1)
        elif self.model_type == "DLinear":
            input_tensor = input_tensor.to(self.model.device)
            target = target.to(self.model.device)
            input_tensor.requires_grad = True
            out = self.model(input_tensor)
            input_tensor.retain_grad()
            loss_func(out, target).backward()
            output = input_tensor.grad.abs()
        elif self.model_type == "TimesNet":
            # compute saliency maps batchwise
            output = []
            times_net_batch_size = 256
            for i in tqdm(range(0, len(input_tensor), times_net_batch_size)):
                batch = input_tensor[i : i + times_net_batch_size]
                batch = batch.to(self.model.device)
                target_batch = target[i : i + times_net_batch_size].to(self.model.device)
                batch.requires_grad = True
                out = self.model(batch)
                batch.retain_grad()
                loss_func(out, target_batch).backward()
                output.append(batch.grad.abs())
            output = torch.cat(output, dim=0)
        elif (
            self.model_type == "VQ-VAE_Transformer"
            or self.model_type == "DVAE_Transformer"
        ):
            output = []
            transformer_batch_size = 256
            for i in tqdm(range(0, len(input_tensor), transformer_batch_size)):
                batch = input_tensor[i : i + transformer_batch_size]
                batch = batch.to(self.model.device)
                target_batch = target[i : i + transformer_batch_size].to(self.model.device)
                x = self.model.embedding(batch)
                # Store original embedding for gradient computation
                orig_x = x
                orig_x.retain_grad()
                # Forward pass through transformer
                for block in self.model.transformer.h:
                    x = block(x)
                x = self.model.transformer.ln_f(x)
                x = self.model.class_head.linear_1(x)
                x = self.model.class_head.activation(x.squeeze(-1))
                x = self.model.class_head.linear_2(x)
                # Compute loss and backprop
                loss_func(x, target_batch).backward()
                batch_output = orig_x.grad.abs().sum(dim=-1)
                output.append(batch_output)
            output = torch.cat(output, dim=0)
        elif self.model_type == "TS_Transformer":
            transformer_batch_size = 256
            output = []
            for i in tqdm(range(0, len(input_tensor), transformer_batch_size)):
                batch = input_tensor[i : i + transformer_batch_size]
                batch = batch.to(self.model.device)
                target_batch = target[i : i + transformer_batch_size].to(self.model.device)
                batch.requires_grad = True
                out = self.model(batch)
                batch.retain_grad()
                loss_func(out, target_batch).backward()
                output.append(batch.grad.abs())
            output = torch.cat(output, dim=0)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        output = output.cpu()
        if save_to_pickle:
            self.save(output, save_path)

        return output
