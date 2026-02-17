import numpy as np
import logging as log
import argparse
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from model.DLinear import DLinear
from model.TimesNet import TimesNet
from model.mlp import MLP
from model.ts_transformer import TS_Transformer
from lightning import Trainer
from params import DATASET_MODULES
from utils import (
    get_logger,
    check_dataset_name,
    get_dataset_module,
    get_data_dim,
    get_model_hash,
    get_num_classes,
    check_model_name,
    load_val_test_idx,
)


def init_model(conf: dict[str, any]):
    log.info("Initializing No Latentspace model with the following configuration:")
    log.info(conf)

    if conf["model_name"] == "DLinear":
        model = DLinear(
            input_size=conf["input_size"],
            in_dim=conf["data_dim"],
            num_class=conf["num_classes"],
            individual=conf["individual"],
            kernel_size=conf["kernel_size"],
            moving_avg=conf["moving_avg"],
            learning_rate=conf["learning_rate"],
            hyperparams_search_str=conf["hyperparams_search_str"],
            dataset_name=conf["dataset"],
            hash_model=get_model_hash(conf),
        )
    elif conf["model_name"] == "TimesNet":
        model = TimesNet(
            input_size=conf["input_size"],
            in_dim=conf["data_dim"],
            num_class=conf["num_classes"],
            d_model=conf["d_model"],
            n_hidden_layers=conf["n_hidden_layers"],
            dropout_p=conf["dropout_p"],
            learning_rate=conf["learning_rate"],
            freq=conf["freq"],
            embed_type=conf["embed_type"],
            d_ff=conf["d_ff"],
            num_kernels=conf["num_kernels"],
            top_k=conf["top_k"],
            hyperparams_search_str=conf["hyperparams_search_str"],
            dataset_name=conf["dataset"],
            hash_model=get_model_hash(conf),
        )
    elif conf["model_name"] == "MLP":
        model = MLP(
            input_size=conf["input_size"],
            num_class=conf["num_classes"],
            in_dim=conf["data_dim"],
            n_hidden_layers=conf["n_hidden_layers"],
            d_model=conf["d_model"],
            dropout_p=conf["dropout_p"],
            learning_rate=conf["learning_rate"],
            use_layer_norm=conf["use_layer_norm"],
            use_latent_input=False,
            num_latent_tokens=0,
            hyperparams_search_str=conf["hyperparams_search_str"],
            dataset_name=conf["dataset"],
            hash_model=get_model_hash(conf),
        )
    elif conf["model_name"] == "TS_Transformer":
        model = TS_Transformer(
            seq_len=conf["input_size"],
            in_dim=conf["data_dim"],
            n_classes=conf["num_classes"],
            d_model=conf["d_model"],
            freq=conf["freq"],
            embed_type=conf["embed_type"],
            n_blocks=conf["n_blocks"],
            n_head=conf["n_head"],
            res_dropout=conf["res_dropout"],
            att_dropout=conf["att_dropout"],
            learning_rate=conf["learning_rate"],
            hyperparams_search_str=conf["hyperparams_search_str"],
            dataset_name=conf["dataset"],
            hash_model=get_model_hash(conf),
        )
    else:
        raise ValueError(f"Model {conf['model_name']} not found")

    return model


def train_model(data_module, conf: dict[str, any]):
    score = "val/loss"
    mode = "min"
    model_name = conf["model_name"]

    early_stop_callback = EarlyStopping(
        monitor=score, min_delta=0.001, patience=10, verbose=False, mode=mode
    )
    model_checkpoint = ModelCheckpoint(
        monitor=score,
        dirpath=f"model_checkpoints/{model_name}",
        filename=f"{model_name}",
        save_top_k=1,
        save_last=True,
        mode=mode,
    )
    mlflow_logger = get_logger(use_mlflow=conf["use_mlflow"], experiment_name=conf["mlflow_experiment_name"])

    # Log the gradient_clip_val
    if conf["use_mlflow"]:
        mlflow_logger.log_hyperparams(
            {
                "gradient_clip_val": conf["gradient_clip_val"],
                "model_name": conf["model_name"],
                "batch_size": conf["batch_size"],
                "max_epochs": conf["epochs"],
                "seed": conf["seed"],
            }
        )

    model = init_model(conf)

    trainer = Trainer(
        accelerator="cpu" if conf["n_gpus"] == 0 else "auto",
        devices=conf["n_gpus"] if conf["n_gpus"] > 1 else 1,
        strategy=DDPStrategy(find_unused_parameters=True) if conf["n_gpus"] > 1 else "auto",
        num_nodes=1,
        logger=mlflow_logger,
        callbacks=[early_stop_callback, model_checkpoint],
        max_epochs=conf["epochs"],
        gradient_clip_val=conf["gradient_clip_val"],
    )

    log.info("Starting training")
    trainer.fit(model, data_module)
    best_val_score = model.best_val_score
    print(f"Best val loss: {best_val_score}")
    log.info("Training finished")
    trainer.test(model, dataloaders=data_module.test_dataloader())
    return best_val_score


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--model-name", type=check_model_name, default="TS_Transformer")
    args.add_argument("--dataset", type=check_dataset_name, default="ECG")
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--n-resblocks", type=int, default=4)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--gradient-clip-val", type=float, default=1.0)
    args.add_argument("--dropout-p", type=float, default=0.1)
    args.add_argument(
        "--hyperparams-search-str", type=str, default="NoHyperparamSearch"
    )
    args.add_argument("--dataset-path", type=str, default="data")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--individual", action=argparse.BooleanOptionalAction)
    args.add_argument("--kernel-size", type=int, default=3)
    args.add_argument("--moving-avg", type=int, default=3)
    args.add_argument("--freq", type=str, default="h")
    args.add_argument("--embed-type", type=str, default="timeF")
    args.add_argument("--d-ff", type=int, default=64)
    args.add_argument("--num-kernels", type=int, default=6)
    args.add_argument("--top-k", type=int, default=2)
    args.add_argument("--d-model", type=int, default=64)
    args.add_argument("--n-head", type=int, default=4)
    args.add_argument("--n-blocks", type=int, default=2)
    args.add_argument("--res-dropout", type=float, default=0.1)
    args.add_argument("--att-dropout", type=float, default=0.0)
    args.add_argument("--n-hidden-layers", type=int, default=4)
    args.add_argument("--input-size", type=int, default=200)
    args.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--mlflow-experiment-name", type=str, default="xai-ts-classification")
    args.add_argument("--n-gpus", type=int, default=1)
    conf = vars(args.parse_args())
    log.info(conf)

    torch.set_float32_matmul_precision("medium")

    # lightning seed everthing
    pl.seed_everything(conf["seed"])

    data_path = Path(conf["dataset_path"])
    conf["num_classes"] = get_num_classes(conf["dataset"])
    conf["data_dim"] = get_data_dim(conf["dataset"])


    if (data_path / conf["dataset"] / "val_idx.npy").exists():
        # params
        val_idx, test_idx = load_val_test_idx(data_path / conf["dataset"])
    else:
        raise ValueError("val_idx.npy or test_idx.npy not found")


    log.info("Loading data")
    data_module: DATASET_MODULES = get_dataset_module(conf["dataset"])(
        dataset_path=data_path / conf["dataset"],
        ds_type="classification",
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        batch_size=conf["batch_size"],
    )

    data_module.setup(seq_len=1)

    log.info("Initialize Training No Latent Model")
    train_model(data_module, conf)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
