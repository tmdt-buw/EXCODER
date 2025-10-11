import argparse
import re
from pathlib import Path
import pickle
import logging
from typing import Any
import numpy as np
from lightning import seed_everything
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from xai_methods.lime import Lime
from model.transformer_decoder import MyTransformerDecoder
from data_loader.laten_ds_helper import create_latent_space_dataset_VQ_VAE_IDs
from data_loader.utils import MyScaler
from data_loader.data_module import SimpleDataModule
from xai_similar_subsequence_accuracy_for_datasets import main as compute_ssa
from utils import (
    load_val_test_idx,
    load_first_stage_model,
    get_dataset_module,
    get_laten_ds
)
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
    XAI_METHODS,
    DATASET_MODULES
)


def load_new_ts_data(path: Path, file_name: str, scaler: MyScaler | StandardScaler):
    """Load and preprocess time series data for inference.

    This function loads time series data and labels from numpy files, applies
    scaling transformation, and creates a PyTorch DataLoader for batch processing.

    Args:
        path (Path): Directory path containing the data files.
        file_name (str): Base name of the data files (without extension).
        scaler (MyScaler | StandardScaler): Fitted scaler for data normalization.

    Returns:
        DataLoader: PyTorch DataLoader containing the preprocessed data and labels.
    """
    data_path = path / f"{file_name}_x.npy"
    label_path = path / f"{file_name}_labels.npy"
    data = np.load(data_path)
    labels = np.load(label_path)
    data = scaler.transform(data)
    torch_ds = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(torch_ds, batch_size=512, shuffle=False)
    return dataloader

def get_scaler(dataset_name: DATASET_NAMES, data_path: Path, project_path: Path):
    """Get the fitted scaler for a specific dataset.

    This function loads the validation and test indices, creates a dataset module,
    sets it up, and returns the fitted scaler used for data normalization.

    Args:
        dataset_name (DATASET_NAMES): Name of the dataset.
        data_path (Path): Path to the data directory.
        project_path (Path): Path to the project root directory.

    Returns:
        MyScaler | StandardScaler: Fitted scaler for the dataset.
    """

    val_idx, test_idx = load_val_test_idx(project_path / data_path / dataset_name)

    data_module: DATASET_MODULES = get_dataset_module(conf["dataset_name"])(
        dataset_path=data_path / conf["dataset_name"],
        ds_type="classification",
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        batch_size=conf["batch_size"]
    )

    data_module.setup(seq_len=1)

    return data_module.scaler

def create_latent_ds(
        data_path: Path,
        dataset_name: DATASET_NAMES,
        new_ds_name: str,
        scaler: MyScaler | StandardScaler,
        device: torch.device | str = "cpu",
        conf: dict[str, Any] = None
    ):
    """Create latent space dataset using VQ-VAE model.

    This function loads new time series data, passes it through a pre-trained
    VQ-VAE model to create latent representations, and adds start tokens for
    autoregressive modeling.

    Args:
        data_path (Path): Path to the data directory.
        dataset_name (DATASET_NAMES): Name of the dataset.
        new_ds_name (str): Name of the new dataset to process.
        scaler (MyScaler | StandardScaler): Fitted scaler for data normalization.
        device (torch.device | str, optional): Device for computation. Defaults to "cpu".
        conf (dict[str, Any], optional): Configuration dictionary. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the latent representations
            with start tokens and the corresponding labels.
    """

    new_dataloader = load_new_ts_data(data_path / dataset_name, new_ds_name, scaler)
    vq_vae_model = load_first_stage_model(Path(conf["path_vq_vae"]))
    num_embeddings = vq_vae_model.num_embeddings
    x_data, labels = create_latent_space_dataset_VQ_VAE_IDs(
        vq_vae_model,
        loader=new_dataloader,
        seq_len=1,
        has_patch_embed=True,
        no_labels=False,
        device=device,
    )
    start_token = num_embeddings + 1
    start_vec = np.full((len(x_data),), fill_value=start_token)
    x_data = np.concatenate([start_vec[:, None], x_data], axis=1)

    return x_data, labels

def model_init(model_path: Path):
    """Initialize and configure a transformer decoder model from checkpoint.

    This function loads a MyTransformerDecoder model from a checkpoint file and
    sets up the necessary hyperparameters and attributes for inference with
    latent input data.

    Args:
        model_path (Path): Path to the model checkpoint file.

    Returns:
        MyTransformerDecoder: Loaded and configured transformer model.
    """
    model = MyTransformerDecoder.load_from_checkpoint(model_path)
    model.hparams["use_latent_input"] = True
    model.input_size = (
        model.input_size if hasattr(model, "input_size") else model.seq_len
    )
    model.in_dim = model.in_dim if hasattr(model, "in_dim") else 1
    model.num_classes = (
        model.num_classes if hasattr(model, "num_classes") else model.n_classes
    )
    model.eval()
    return model


def predict_model(model: MyTransformerDecoder, dataset: torch.Tensor):
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    predictions = []
    for batch in dataloader:
        batch = batch.to(model.device)
        logits = model(batch, generate=False)
        predictions.append(logits.argmax(dim=1))
    return torch.cat(predictions).cpu()



def main(conf: dict[str, any]):
    """Main function to generate XAI explanations for time series models.

    This function orchestrates the entire pipeline for generating explainable AI
    explanations on time series data. It loads data, creates latent representations,
    initializes the model, and generates explanations using the specified XAI method.

    Args:
        conf (dict[str, any]): Configuration dictionary containing:
            - seed (int): Random seed for reproducibility
            - dataset_name (str): Name of the dataset to process
            - model_type (str): Type of model to use
            - xai_method (str): XAI method to apply
            - data_path (str): Path to the data directory
            - new_ds_name (str): Name of the new dataset
            - lime_num_samples (int): Number of samples for LIME explanations
            - batch_size (int): Batch size for data loading
    """
    seed_everything(conf["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name: DATASET_NAMES = conf["dataset_name"]
    model_type: MODEL_NAMES = conf["model_type"]
    xai_method: XAI_METHODS = conf["xai_method"]

    if not xai_method == "LIME":
        raise ValueError(f"XAI method {xai_method} not supported")

    logging.info(f"Running {xai_method} on {dataset_name} with {model_type}")

    data_path = Path(conf["data_path"])
    project_path = data_path.parent
    output_path = (
        data_path
        / "XAI_Results"
        / dataset_name
        / model_type
        / f"seed_{conf['seed']}"
        / f"{xai_method}_{conf['new_ds_name']}_explanations.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_ssa = output_path.parent / f"ssa_results_{conf['new_ds_name']}.pkl"

    if output_path_ssa.exists():
        logging.info(f"SSA results already exist for {conf['new_ds_name']}")
        return
    model_path = (
        project_path
        / "model_checkpoints/best_models"
        / dataset_name
        / model_type
        / f"seed_{conf['seed']}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output path: {output_path}")

    checkpoint_files = list(model_path.glob("*.ckpt"))
    model_path = checkpoint_files[0]
    scaler = get_scaler(dataset_name, data_path, project_path)

    x_data, labels = create_latent_ds(data_path, dataset_name, conf["new_ds_name"], scaler, device, conf)
    model = model_init(model_path)

    
    if output_path.exists():
        logging.info(f"Loading LIME explanations from {output_path}")
        with open(output_path, "rb") as f:
            lime_explanations = pickle.load(f)
    else:
        logging.info(f"Creating LIME explanations for {output_path}")
        lime_class_names = ["Bad", "Good"]
        logging.info(
            f"LIME dataset shape: {x_data.shape} dtype: {x_data.dtype}"
        )
        lime = Lime(
            model=model,
            dataset=x_data,
            class_names=lime_class_names,
            num_samples=conf["lime_num_samples"],
            use_latent_input=model.hparams.get("use_latent_input", False),
            model_type=model_type,
            dataset_type=dataset_name,
            discretize_continuous=False,
            verbose=False,
            conf=conf,
        )
        lime_explanations, _ = lime.explain(
            x_data,
            save_to_pickle=True,
            save_path=str(output_path),
        )
        lime_explanations = lime_explanations.reshape(
            -1, model.input_size, model.in_dim
        )
        logging.info(f"LIME explanations shape: {lime_explanations.shape}")
        #save the explanations to a pickle file
        with open(output_path, "wb") as f:
            pickle.dump(lime_explanations, f)

    (
        _,_,_,
        class_train_ds,
        class_val_ds,
        class_test_ds,
    ) = get_laten_ds(
        vq_vae_path=conf["path_vq_vae"],
        dataset_name=conf["dataset"],
        init_ds=False,
        n_cycles=1,
        data_path=data_path,
        prob_unk_token=0.0,
    )

    data_module = SimpleDataModule(
        train_ds=class_train_ds,
        val_ds=class_val_ds,
        test_ds=class_test_ds,
        batch_size=conf["batch_size"],
    )

    train_ds = data_module.train_dataloader().dataset
    train_dataset_size = len(train_ds)
    train_ds_x = torch.stack([train_ds[i][0] for i in range(train_dataset_size)])
    train_ds_y = torch.stack([train_ds[i][1] for i in range(train_dataset_size)])

    model_predictions = predict_model(model, x_data)
    logging.info(f"Model predictions shape: {model_predictions.shape}")

    conf["subseq_len"] = 1
    conf["min_amount_similar_subseq"] = 20
    logging.info(f"Computing SSA for {conf['new_ds_name']}")
    ssa_results = compute_ssa(
        conf=conf,
        dataset=torch.tensor(x_data, dtype=torch.long),
        target=torch.tensor(labels, dtype=torch.long),
        explanations=lime_explanations.reshape(lime_explanations.shape[0], -1),
        model_predictions=model_predictions,
        train_dataset=train_ds_x,
        train_target=train_ds_y,
        top_k=3
    )

    # save ssa results to a pickle file
    with open(output_path_ssa, "wb") as f:
        pickle.dump(ssa_results, f)

    return ssa_results


def get_data_files(data_path: Path):
    """
    Get the data files from the data path.
    The data files are the files that end with _x.npy and are in the format {number}_{number}_x.npy
    The function returns the names of the data files without the _x.npy extension.

    Args:
        data_path (Path): The path to the data directory.

    Returns:
        list[str]: The names of the data files without the _x.npy extension.
    """
    all_npy_files = list(data_path.glob("*.npy"))
    pattern = re.compile(r'^\d+_\d+_x\.npy$')
    data_files = [f for f in all_npy_files if pattern.match(f.name)]
    data_files = [f.stem.replace("_x", "") for f in data_files]
    return data_files


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = argparse.ArgumentParser()
    args.add_argument("--dataset-name", type=str, default="Welding")
    args.add_argument("--model-type", type=str, default="VQ-VAE_Transformer")
    args.add_argument("--xai-method", type=str, default="LIME")
    args.add_argument("--data-path", type=str, default="data")
    args.add_argument("--lime-num-samples", type=int, default=1000)
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--new-ds-name", type=str, default="1_28")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--n-cycles", type=int, default=1)
    args.add_argument("--path-vq-vae", type=str, default="model_checkpoints/best_models/VQ-VAE_Welding.ckpt")
    conf = vars(args.parse_args())
    conf["path_vq_vae"] = Path(conf["path_vq_vae"])
    conf["dataset"] = conf["dataset_name"]



    new_ds_names = get_data_files(Path(conf["data_path"]) / conf["dataset_name"])
    print(new_ds_names)

    for new_ds_name in new_ds_names:
        conf["new_ds_name"] = new_ds_name
        logging.info(f"Configuration: {conf}")
        main(conf)
