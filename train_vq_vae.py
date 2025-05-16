import numpy as np
import logging as log
import argparse
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model.vq_vae_patch_embed import VQVAEPatch
from lightning import Trainer
from utils import (
    DATASET_MODULES, 
    get_logger, 
    check_dataset_name, 
    get_dataset_module, 
    get_data_dim, 
    get_model_hash, 
    check_model_name,
    load_val_test_idx
)


def init_model(conf: dict[str, any]):
    log.info("Initializing VQ-VAE model with the following configuration:")
    log.info(conf)

    vq_vae_model = VQVAEPatch(
        hidden_dim=conf["hidden_dim"],
        input_dim=conf["data_dim"],
        num_embeddings=conf["num_embeddings"],
        embedding_dim=conf["embedding_dim"],
        n_resblocks=conf["n_resblocks"],
        patch_size=25,
        seq_len=200,
        kmeans_iters=5,
        threshold_ema_dead_code=2,
        batch_norm=False,
        learning_rate=conf["learning_rate"],
        hyperparams_search_str=conf["hyperparams_search_str"],
        dataset_name=conf["dataset"],
        hash_model=get_model_hash(conf),
        use_dvae_vq=conf["use_dvae_vq"],
        dvae_temperature=conf["dvae_temperature"],
    )

    return vq_vae_model


def train_vq_vae(data_module, conf: dict[str, any]):
    score = "val/loss"
    mode = "min"

    early_stop_callback = EarlyStopping(
        monitor=score, min_delta=0.001, patience=10, verbose=False, mode=mode
    )
    model_checkpoint = ModelCheckpoint(
        monitor=score,
        dirpath="model_checkpoints/VQ-VAE",
        filename="vq_vae",
        save_top_k=1,
        save_last=True,
        mode=mode,
    )
    mlflow_logger = get_logger(use_mlflow=conf["use_mlflow"], experiment_name=conf["mlflow_experiment_name"])

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

    vq_vae_model = init_model(conf)

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
    trainer.fit(vq_vae_model, data_module)
    best_val_loss = vq_vae_model.best_val_loss
    print(f"Best val loss: {best_val_loss}")
    log.info("Training finished")
    trainer.test(vq_vae_model, dataloaders=data_module.test_dataloader())
    return best_val_loss


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--model-name", type=check_model_name, default="VQ-VAE")
    args.add_argument("--dataset", type=check_dataset_name, default="CNC_Machining")
    args.add_argument("--hidden-dim", type=int, default=256)
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--patch-size", type=int, default=25)
    args.add_argument("--num-embeddings", type=int, default=256)
    args.add_argument("--embedding-dim", type=int, default=32)
    args.add_argument("--n-resblocks", type=int, default=4)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--gradient-clip-val", type=float, default=1.0)
    args.add_argument("--dropout-p", type=float, default=0.1)
    args.add_argument("--hyperparams-search-str", type=str, default="NoHyperparamSearch")
    args.add_argument("--dataset-path", type=str, default="data")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--mlflow-experiment-name", type=str, default="xai-ts-classification")
    args.add_argument("--dvae-temperature", type=float, default=1.0)
    args.add_argument("--n-gpus", type=int, default=1)
    conf = vars(args.parse_args())
    log.info(conf)

    torch.set_float32_matmul_precision("medium")

    # lightning seed everthing
    pl.seed_everything(conf["seed"])

    data_path = Path(conf["dataset_path"])

    # params    
    if not (data_path / conf["dataset"] / "val_idx.npy").exists() or not (data_path / conf["dataset"] / "test_idx.npy").exists():
        val_idx = None
        test_idx = None 
        log.warning("val_idx.npy or test_idx.npy not found, using train_val_test_split")
    else:
        val_idx, test_idx = load_val_test_idx(data_path / conf["dataset"])

    log.info("Loading data")
    data_module: DATASET_MODULES = get_dataset_module(conf["dataset"])(
        dataset_path=data_path / conf["dataset"],
        ds_type="reconstruction",
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        batch_size=conf["batch_size"]
    )

    data_module.setup(seq_len=1)

    # save val_idx and test_idx
    if not (data_path / conf["dataset"] / "val_idx.npy").exists():
        np.save(f"{data_path / conf['dataset']}/val_idx.npy", data_module.val_idx)
    if not (data_path / conf["dataset"] / "test_idx.npy").exists():
        np.save(f"{data_path / conf['dataset']}/test_idx.npy", data_module.test_idx)

    log.info("Initialize Training VQ-VAE")
    conf["data_dim"] = get_data_dim(conf["dataset"])
    conf["use_dvae_vq"] = conf["model_name"] == "DVAE"
    train_vq_vae(data_module, conf)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
