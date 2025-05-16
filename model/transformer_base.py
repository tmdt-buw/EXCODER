import torch
import math
import numpy as np
from torch import nn, optim
import logging
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score as f1
import lightning.pytorch as pl
from model.embedding import LatentEmbedding
from model.transformer_block import Block
from abc import ABC, abstractmethod
from torchmetrics import Accuracy, F1Score


class TransformerDecoderBase(pl.LightningModule):
    """
    Base class for transformer decoder models using PyTorch Lightning.
    
    Implements common functionality including training loop, metrics tracking,
    and optimization setup for both generation and classification tasks.
    """

    def __init__(
        self,
        dataset_name: str,
        d_model: int = 64,
        embedding_classes: int = 131,
        seq_len: int = 100,
        n_blocks: int = 2,
        n_head: int = 6,
        res_dropout=0.1,
        att_dropout=0.0,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        class_h_bias: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.d_model = d_model
        self.embedding_classes = embedding_classes
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.n_head = n_head
        self.res_dropout = res_dropout
        self.att_dropout = att_dropout
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.class_h_bias = class_h_bias
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1
        
        self.best_val_score = 0
        task = "binary" if n_classes == 2 else "multiclass"

        self.train_accuracy = Accuracy(task=task, num_classes=n_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.val_accuracy = Accuracy(task=task, num_classes=n_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.test_accuracy = Accuracy(task=task, num_classes=n_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.save_hyperparameters()

    def _init_weights(self, module):
        """
        Initialize the weights of the model's modules.

        Args:
            module: The module whose weights to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @abstractmethod
    def forward(self, x: torch.Tensor, generate: bool = True) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor
            generate (bool): Whether to use generation or classification head

        Returns:
            torch.Tensor: Model output
        """
        raise NotImplementedError

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute the loss for the current task.

        Args:
            batch (tuple): A tuple containing the input data and labels.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        if self.task == "generate":
            return self.step_task_gen(batch)
        elif self.task == "classification":
            return self.step_task_class(batch)

    @abstractmethod
    def step_task_gen(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass for generation task.
        
        Args:
            batch (tuple): Input batch
            
        Returns:
            tuple: (loss, logits, labels)
        """
        raise NotImplementedError

    @abstractmethod
    def step_task_class(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass for classification task.
        
        Args:
            batch (tuple): Input batch
            
        Returns:
            tuple: (loss, logits, labels)
        """
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning training step"""
        loss, logits, labels = self._step(batch)
        if self.task == "generate":
            self.log("train/loss", loss.item(), prog_bar=True)
        else:
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)
            acc = self.train_accuracy(preds, labels)
            f1score = self.train_f1(preds, labels)
            self.log("train/acc", acc.item(), prog_bar=True)
            self.log("train/f1_score", f1score.item(), prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning validation step"""
        loss, logits, labels = self._step(batch)
        self.log("val/loss", loss.item(), prog_bar=True, sync_dist=True)
        if self.task == "classification":
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)
            _ = self.val_accuracy(preds, labels)
            _ = self.val_f1(preds, labels)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning test step"""
        loss, logits, labels = self._step(batch)
        self.log("test/loss", loss.item(), prog_bar=True, sync_dist=True)
        if self.task == "classification":
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)
            _ = self.test_accuracy(preds, labels)
            _ = self.test_f1(preds, labels)
        return loss

    def on_validation_epoch_start(self):
        """PyTorch Lightning validation epoch start hook"""
        if self.task == "classification":
            self.val_accuracy.reset()
            self.val_f1.reset()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        """PyTorch Lightning validation epoch end hook"""
        if self.task == "classification":
            val_acc = self.val_accuracy.compute()
            val_f1 = self.val_f1.compute()
            self.log("val/f1_score", val_f1.item(), sync_dist=True, prog_bar=True)
            self.log("val/acc", val_acc.item(), sync_dist=True, prog_bar=True)
            if val_f1 > self.best_val_score:
                self.best_val_score = val_f1
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        """PyTorch Lightning test epoch start hook"""
        if self.task == "classification":
            self.test_accuracy.reset()
            self.test_f1.reset()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        """PyTorch Lightning test epoch end hook"""
        if self.task == "classification":   
            test_acc = self.test_accuracy.compute()
            test_f1 = self.test_f1.compute()
            self.log("test/f1_score", test_f1.item(), sync_dist=True, prog_bar=True)
            self.log("test/acc", test_acc.item(), sync_dist=True, prog_bar=True)
        return super().on_test_epoch_end()

    def loss_cross_entropy(self, logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """
        Compute the loss for the cross entropy task.
        """
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1), 
            ignore_index=ignore_index
        )

    def loss_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the mse task.
        """
        return F.mse_loss(pred, target)

    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns:
            torch.optim.Optimizer: The configured optimizer
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=self.learning_rate, betas=self.betas)
        optimizer = torch.optim.RAdam(
            optim_groups, lr=self.learning_rate, betas=self.betas
        )

        return optimizer

    def switch_to_generate(self):
        """
        Switch the model to generate mode. This disables the classification head for the forward pass.
        """
        self.task = "generate"

    def switch_to_classification(self):
        """
        Switch the model to classification mode. This enables the classification head for the forward pass.
        """
        self.task = "classification"
