from pathlib import Path
import os
from typing import Literal
from mlflow_logger import CustomLightningMLFlowLogger
from data_loader.data_module import WeldingDataModule, CNCDataModule, ECGDataModule
from data_loader.dataset import (
    ClassificationDataset,
    MyLatentClassificationDataset,
    MyLatentAutoregressiveDataset,
)

from argparse import ArgumentTypeError
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loader.utils import save_all_ds_ids, load_all_ds_ids
from data_loader.laten_ds_helper import create_autoreg_ds
from model.vq_vae_patch_embed import VQVAEPatch
from model.mlp import MLP
from model.DLinear import DLinear
from model.TimesNet import TimesNet
from model.transformer_decoder import MyTransformerDecoder
from model.ts_transformer import TS_Transformer
from model.sax_representation import SAX
from data_loader.data_module import SimpleDataModule
from params import DATASET_NAMES, DATASET_MODULES, MODEL_NAMES


def check_dataset_name(arg_value: str) -> str:
    """
    Validate that the dataset name is one of the allowed values.

    Args:
        arg_value (str): Name of dataset to validate

    Returns:
        str: Validated dataset name

    Raises:
        ArgumentTypeError: If dataset name is invalid
    """
    valid_names = tuple(DATASET_NAMES.__args__)  # Convert to tuple for joining
    if arg_value not in valid_names:
        raise ArgumentTypeError(
            f"Invalid dataset name: {arg_value}. Choose from: {', '.join(valid_names)}"
        )
    return arg_value


def check_model_name(arg_value: str) -> str:
    """
    Validate that the model name is one of the allowed values.

    Args:
        arg_value (str): Name of model to validate

    Returns:
        str: Validated model name

    Raises:
        ArgumentTypeError: If model name is invalid
    """
    valid_names = tuple(MODEL_NAMES.__args__)  # Convert to tuple for joining
    if arg_value not in valid_names:
        raise ArgumentTypeError(
            f"Invalid model name: {arg_value}. Choose from: {', '.join(valid_names)}"
        )
    return arg_value


def get_dataset_module(dataset_name: DATASET_NAMES) -> type[DATASET_MODULES]:
    """
    Get the appropriate dataset module class for the given dataset name.

    Args:
        dataset_name (DATASET_NAMES): Name of the dataset

    Returns:
        type[DATASET_MODULES]: The dataset module class

    Raises:
        ValueError: If dataset name is invalid
    """
    if dataset_name == "Welding":
        return WeldingDataModule
    elif dataset_name == "CNC_Machining":
        return CNCDataModule
    elif dataset_name == "ECG":
        return ECGDataModule
    elif dataset_name == "UEA":
        raise NotImplementedError("UEA dataset not implemented")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def get_data_dim(dataset_name: DATASET_NAMES) -> int:
    """
    Get the data dimension for the specified dataset.

    Args:
        dataset_name (DATASET_NAMES): Name of the dataset

    Returns:
        int: Number of dimensions in the dataset

    Raises:
        ValueError: If dataset name is invalid
    """
    if dataset_name == "ECG":
        return 1
    elif dataset_name == "CNC_Machining":
        return 3
    elif dataset_name == "Welding":
        return 2
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def get_num_classes(dataset_name: DATASET_NAMES) -> int:
    if dataset_name == "ECG":
        return 5
    elif dataset_name == "CNC_Machining":
        return 2
    elif dataset_name == "Welding":
        return 2
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def get_logger(use_mlflow: bool = True, experiment_name: str = "xai-ts-classification"):
    """
    Get the appropriate logger based on configuration.

    Args:
        use_mlflow (bool): Whether to use MLflow logger

    Returns:
        Logger: MLflow logger if use_mlflow is True, else None
    """
    if use_mlflow:
        logger = CustomLightningMLFlowLogger(
            experiment_name=experiment_name, log_model=True
        )
    else:
        logger = None

    return logger


def get_model_hash(params: dict[str, any]) -> str:
    """
    Creates a unique hash based on model parameters.

    Returns:
        str: A unique hash string representing the model configuration
    """
    # Create a sorted string representation of parameters
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))

    # Use hash function to create a unique identifier
    return hashlib.md5(param_str.encode()).hexdigest()


def load_val_test_idx(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load validation and test indices from numpy files.

    Args:
        data_path (Path): Path to directory containing val_idx.npy and test_idx.npy

    Returns:
        tuple: (validation indices array, test indices array)
    """
    val_idx = np.load(f"{data_path}/val_idx.npy")
    test_idx = np.load(f"{data_path}/test_idx.npy")

    return val_idx, test_idx


def load_first_stage_model(model_path: Path):
    model_name = model_path.stem.split("_")[0]
    if model_name == "VQ-VAE" or model_name == "DVAE":
        model = VQVAEPatch.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def get_laten_ds(
    vq_vae_path: Path,
    dataset_name: DATASET_NAMES,
    init_ds: bool = True,
    data_path: str = "data",
    n_cycles: int = 1,
    prob_unk_token: float = 0.0,
    seq_prediction_task: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(data_path)

    if seq_prediction_task:
        data_id_path = data_path / f"ds_latent_ids_{vq_vae_path.stem}"
    else:
        data_id_path = data_path / f"ds_latent_ids_{vq_vae_path.stem}_CO"

    vq_vae_model = load_first_stage_model(vq_vae_path)
    vq_vae_model.eval()

    if not data_id_path.exists() or init_ds:
        val_idx, test_idx = load_val_test_idx(data_path / dataset_name)

        data_module: DATASET_MODULES = get_dataset_module(dataset_name)(
            dataset_path=data_path / dataset_name,
            ds_type="reconstruction",
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
        )
        recon_train_ds, recon_val_ds, recon_test_ds = create_autoreg_ds(
            vq_vae_model,
            data_module,
            task="reconstruction",
            seq_len=n_cycles,
            device=device,
            prob_unk_token=prob_unk_token,
        )

        data_module: DATASET_MODULES = get_dataset_module(dataset_name)(
            dataset_path=data_path / dataset_name,
            ds_type="classification",
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
        )
        class_train_ds, class_val_ds, class_test_ds = create_autoreg_ds(
            vq_vae_model,
            data_module,
            task="classification",
            seq_len=n_cycles,
            device=device,
            prob_unk_token=prob_unk_token,
            seq_prediction_task=seq_prediction_task,
        )

        save_all_ds_ids(
            recon_train_ds=recon_train_ds,
            recon_val_ds=recon_val_ds,
            recon_test_ds=recon_test_ds,
            class_train_ds=class_train_ds,
            class_val_ds=class_val_ds,
            class_test_ds=class_test_ds,
            path=data_id_path,
        )
    else:
        (
            recon_train_ds,
            recon_val_ds,
            recon_test_ds,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = load_all_ds_ids(path=data_id_path)

    return (
        recon_train_ds,
        recon_val_ds,
        recon_test_ds,
        class_train_ds,
        class_val_ds,
        class_test_ds,
    )


def get_classification_models_and_data(
    dataset_name: str,
    model_type: str,
    batch_size: int = 512,
    data_path: str = "data",
    seed: int = 0,
):
    project_path = Path(os.path.abspath(""))

    # Convert paths to Path objects consistently
    model_path = (
        project_path
        / "model_checkpoints/best_models"
        / dataset_name
        / model_type
        / f"seed_{seed}"
    )

    # find all the ckpt files in the model_path
    checkpoint_files = list(model_path.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    model_path = checkpoint_files[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if model_type == "MLP" or model_type == "VQ-VAE_MLP" or model_type == "DVAE_MLP" or model_type == "SAX_MLP":
        model = MLP.load_from_checkpoint(model_path)
    elif model_type == "DLinear":
        model = DLinear.load_from_checkpoint(model_path)
    elif model_type == "TimesNet":
        model = TimesNet.load_from_checkpoint(model_path)
    elif model_type == "VQ-VAE_Transformer" or model_type == "DVAE_Transformer":
        model = MyTransformerDecoder.load_from_checkpoint(model_path)
    elif model_type == "TS_Transformer":
        model = TS_Transformer.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")

    model.eval()

    # Fix path concatenation
    data_path = Path(data_path)  # Convert to Path object
    val_idx, test_idx = load_val_test_idx(project_path / data_path / dataset_name)

    data_id_path = (
        project_path
        / data_path
        / f"ds_latent_ids_{model_type.split('_')[0]}_{dataset_name}"
    )
    if model_type.startswith("VQ-VAE_") or model_type.startswith("DVAE_"):
        (
            _,
            _,
            _,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = load_all_ds_ids(path=data_id_path)

        data_module = SimpleDataModule(
            train_ds=class_train_ds,
            val_ds=class_val_ds,
            test_ds=class_test_ds,
            batch_size=batch_size,
        )
    elif model_type.startswith("SAX_"):
        codebook_size = model.num_latent_tokens - 3
        data_module: DATASET_MODULES = get_dataset_module(dataset_name)(
            dataset_path=project_path / data_path / dataset_name,
            ds_type="classification",
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            batch_size=batch_size,
        )
        class_train_ds, class_val_ds, class_test_ds = convert_to_sax(
            data_module=data_module,
            codebook_size=codebook_size,
            prob_unk_token=0.0,
            task="classification",
        )

        data_module = SimpleDataModule(
            train_ds=class_train_ds,
            val_ds=class_val_ds,
            test_ds=class_test_ds,
            batch_size=batch_size,
        )

    else:
        data_module: DATASET_MODULES = get_dataset_module(dataset_name)(
            dataset_path=project_path / data_path / dataset_name,
            ds_type="classification",
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            batch_size=batch_size,
        )
        data_module.setup(seq_len=1)
    return model, data_module


def get_classification_models_and_data_from_subfolder(
    dataset_name: str,
    model_type: str,
    batch_size: int = 512,
    data_path: str = "data",
    seed: int = 0,
):
    project_path = Path(os.path.abspath("")).resolve().parent

    # Convert paths to Path objects consistently
    model_path = (
        project_path
        / "model_checkpoints/best_models"
        / dataset_name
        / model_type
        / f"seed_{seed}"
    )

    # find all the ckpt files in the model_path
    checkpoint_files = list(model_path.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    model_path = checkpoint_files[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if model_type == "MLP" or model_type == "VQ-VAE_MLP" or model_type == "DVAE_MLP":
        model = MLP.load_from_checkpoint(model_path)
    elif model_type == "DLinear":
        model = DLinear.load_from_checkpoint(model_path)
    elif model_type == "TimesNet":
        model = TimesNet.load_from_checkpoint(model_path)
    elif model_type == "VQ-VAE_Transformer" or model_type == "DVAE_Transformer":
        model = MyTransformerDecoder.load_from_checkpoint(model_path)
    elif model_type == "TS_Transformer":
        model = TS_Transformer.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")

    model.eval()

    # Fix path concatenation
    data_path = Path(data_path)  # Convert to Path object
    val_idx, test_idx = load_val_test_idx(project_path / data_path / dataset_name)

    data_id_path = (
        project_path
        / data_path
        / f"ds_latent_ids_{model_type.split('_')[0]}_{dataset_name}"
    )
    if model_type.startswith("VQ-VAE_") or model_type.startswith("DVAE_"):
        (
            _,
            _,
            _,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = load_all_ds_ids(path=data_id_path)

        data_module = SimpleDataModule(
            train_ds=class_train_ds,
            val_ds=class_val_ds,
            test_ds=class_test_ds,
            batch_size=batch_size,
        )
    else:
        data_module: DATASET_MODULES = get_dataset_module(dataset_name)(
            dataset_path=project_path / data_path / dataset_name,
            ds_type="classification",
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            batch_size=batch_size,
        )
        data_module.setup(seq_len=1)
    return model, data_module


def create_new_datasets(data_path: str):

    dataset_names = ["Welding", "CNC_Machining", "ECG"]
    embedding_model_names = ["VQ-VAE", "DVAE"]
    seq_prediction_tasks = [True, False]
    data_path = Path(data_path)
    for dataset_name in dataset_names:
        for embedding_model_name in embedding_model_names:
            for seq_prediction_task in seq_prediction_tasks:
                vq_vae_path = Path(
                    f"model_checkpoints/best_models/{embedding_model_name}_{dataset_name}.ckpt"
                )
                _ = get_laten_ds(
                    vq_vae_path=vq_vae_path,
                    dataset_name=dataset_name,
                    init_ds=True,
                    n_cycles=1,
                    data_path=data_path,
                    prob_unk_token=0.0,
                    seq_prediction_task=seq_prediction_task,
                )
                print(
                    f"Created new datasets for {dataset_name} with {embedding_model_name} and seq_prediction_task={seq_prediction_task}"
                )


def convert_to_sax(
    data_module: DATASET_MODULES,
    codebook_size: int = 256,
    patch_size: int = 25,
    prob_unk_token: float = 0.0,
    task: Literal[
        "classification", "reconstruction", "autoregressive_classification"
    ] = "classification",
) -> tuple[
    MyLatentAutoregressiveDataset | MyLatentClassificationDataset,
    MyLatentAutoregressiveDataset | MyLatentClassificationDataset,
    MyLatentAutoregressiveDataset | MyLatentClassificationDataset,
]:
    """
    Convert time series data from a data module to Symbolic Aggregate approXimation (SAX) representation.

    This function extracts training, validation, and test datasets from the provided data module,
    transforms them using SAX representation, and returns new ClassificationDataset objects
    containing the transformed data.

    Args:
        data_module (DATASET_MODULES): The data module containing the datasets to transform
        codebook_size (int, optional): The alphabet size for SAX transformation. Defaults to 256.
        patch_size (int, optional): The size of patches to use when calculating word size. Defaults to 25.
        prob_unk_token (float, optional): The probability of an unknown token. Defaults to 0.0.

    Returns:
        tuple[ClassificationDataset, ClassificationDataset, ClassificationDataset]: A tuple containing
            the transformed training, validation, and test datasets as ClassificationDataset objects.
    """

    data_module.setup()
    train_ds: ClassificationDataset = data_module.train_dataloader().dataset
    val_ds: ClassificationDataset = data_module.val_dataloader().dataset
    test_ds: ClassificationDataset = data_module.test_dataloader().dataset

    train_x = train_ds.ds
    val_x = val_ds.ds
    test_x = test_ds.ds

    word_size = train_x.shape[1] // patch_size * train_x.shape[2]

    sax = SAX(word_size=word_size, alphabet_size=codebook_size)
    train_x_sax = sax.transform(train_x)
    val_x_sax = sax.transform(val_x)
    test_x_sax = sax.transform(test_x)

    if task == "classification":
        train_y = train_ds.labels
        val_y = val_ds.labels
        test_y = test_ds.labels
        train_ds = MyLatentClassificationDataset(
            train_x_sax.numpy(), train_y, prob_missing=prob_unk_token
        )
        val_ds = MyLatentClassificationDataset(val_x_sax.numpy(), val_y)
        test_ds = MyLatentClassificationDataset(test_x_sax.numpy(), test_y)
    elif task == "reconstruction":
        train_ds = MyLatentAutoregressiveDataset(
            train_x_sax.numpy(), prob_missing=prob_unk_token
        )
        val_ds = MyLatentAutoregressiveDataset(val_x_sax.numpy())
        test_ds = MyLatentAutoregressiveDataset(test_x_sax.numpy())
    elif task == "autoregressive_classification":
        train_y = train_ds.labels
        val_y = val_ds.labels
        test_y = test_ds.labels
        train_ds = MyLatentAutoregressiveDataset(
            train_x_sax.numpy(), train_y, prob_missing=prob_unk_token
        )
        val_ds = MyLatentAutoregressiveDataset(val_x_sax.numpy(), val_y)
        test_ds = MyLatentAutoregressiveDataset(test_x_sax.numpy(), test_y)
    else:
        raise ValueError(f"Unknown task: {task}")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    create_new_datasets("data_docker")
