import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
import lightning.pytorch as pl
import numpy as np
from abc import abstractmethod
from params import DATASET_NAMES


class ClassificationLightningModule(pl.LightningModule):
    """
    Base class for classification models using PyTorch Lightning.

    Implements common classification functionality including training loop,
    metrics tracking (accuracy, F1 score), and optimization setup.

    Args:
        input_size (int): Size of input features
        num_classes (int): Number of output classes
        in_dim (int): Input dimension
        d_model (int): Size of hidden layers
        n_hidden_layers (int, optional): Number of hidden layers. Defaults to 4
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
        learning_rate (float, optional): Learning rate. Defaults to 1e-3
        hyperparams_search_str (str, optional): String for hyperparameter search. Defaults to "NoHyperparamSearch"
        dataset_name (DATASET_NAMES, optional): Name of the dataset. Defaults to "Welding"
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        in_dim: int,
        d_model: int,
        n_hidden_layers: int = 4,
        dropout_p: float = 0.1,
        learning_rate: float = 1e-3,
        hyperparams_search_str: str = "NoHyperparamSearch",
        dataset_name: DATASET_NAMES = "Welding",
        hash_model: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_hidden_layers = n_hidden_layers
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate

        self.best_val_score = 0
        task = "binary" if num_classes == 2 else "multiclass"

        self.val_losses = []
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")


        self.hyperparams_search_str = hyperparams_search_str
        self.dataset_name = dataset_name
        self.hash_model = hash_model
        self.save_hyperparameters()

    @abstractmethod
    def forward(
        self, x: torch.Tensor   
    ) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Reconstruction loss, data reconstruction, perplexity
        """
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Loss function

        Args:
            logits (torch.Tensor): Logits
            labels (torch.Tensor): Labels

        Returns:
            torch.Tensor: Loss value
        """
        return F.cross_entropy(logits, labels)

    def _get_preds(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(x)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        loss = self.loss(logits, y)
        return preds, loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        PyTorch Lightning calls this inside the training loop
        """
        x, y = batch
        preds, loss = self._get_preds(x, y)
        acc = self.train_accuracy(preds, y)
        f1score = self.train_f1(preds, y)
        if batch_idx % 50 == 0:
            self.log("train/loss", loss.item())
            self.log("train/acc", acc.item())
            self.log("train/f1_score", f1score.item(), prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        PyTorch Lightning calls this inside the validation loop
        """
        x, y = batch
        preds, loss = self._get_preds(x, y)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)
        self.val_losses.append(loss.item())
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        PyTorch Lightning calls this inside the test loop
        """
        x, y = batch
        preds, loss = self._get_preds(x, y)
        self.test_accuracy(preds, y)
        self.test_f1(preds, y)
        return loss

    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_f1.reset()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        val_loss = np.mean(self.val_losses)
        self.log("val/f1_score", val_f1, sync_dist=True, prog_bar=True)
        self.log("val/acc", val_acc, sync_dist=True, prog_bar=True)
        self.log("val/loss", val_loss, sync_dist=True, prog_bar=True)
        if val_f1 > self.best_val_score:
            self.best_val_score = val_f1
        self.val_losses = []
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        test_acc = self.test_accuracy.compute()
        test_f1 = self.test_f1.compute()

        self.log("test/f1_score", test_f1, sync_dist=True, prog_bar=True)
        self.log("test/acc", test_acc, sync_dist=True, prog_bar=True)
        return super().on_test_epoch_end()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.RAdam(self.parameters(), lr=self.learning_rate)
        return optimizer
