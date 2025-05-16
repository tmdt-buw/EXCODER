from typing import Literal
import logging as log
import argparse
import torch
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import seed_everything

from model.tsvq_transformer import TSVQTransformer

from lightning import Trainer
from pathlib import Path
from data_loader.data_module import SimpleDataModule
from data_loader.utils import save_all_ds_ids, load_all_ds_ids
from utils import (
    get_logger,
    check_dataset_name,
    DATASET_NAMES,
    DATASET_MODULES,
    get_dataset_module,
    get_num_classes,
    get_data_dim,
    load_val_test_idx,
)


def init_transformer_model(conf: dict[str, any]):
    log.info("Initializing Transformer model with the following configuration:")
  
    seq_len = (conf["data_dim"] * 8) + 1
    transformer = TSVQTransformer(
        dataset_name=conf["dataset"],
        d_model=conf["d_model"],
        embedding_classes=conf["embedding_classes"],
        seq_len=seq_len,
        data_dim=conf["data_dim"],
        n_blocks=conf["n_resblocks"],
        n_head=conf["n_heads"],
        n_classes=conf["num_classes"],
        res_dropout=conf["res_dropout"],
        att_dropout=conf["att_dropout"],
        learning_rate=conf["learning_rate"],
        class_h_bias=conf["use_class_head_bias"],
        patch_size=25,
        vq_hidden_dim=conf["vq_hidden_dim"],
        vq_n_resblocks=conf["vq_n_resblocks"],
        beta=conf["beta"],
        theta=conf["theta"],
        gamma=conf["gamma"],
    )

    return transformer


def train_transformer(
    model, data_module, conf: dict[str, any], ds_type: Literal["classification", "reconstruction"]
):
    logger = get_logger(use_mlflow=conf["use_mlflow"], experiment_name=conf["mlflow_experiment_name"])

    if ds_type == "classification":
        score = "val/f1_score"
        mode = "max"
    else:
        score = "val/loss"
        mode = "min"
    model_name = conf["model_name"]

    early_stop_callback = EarlyStopping(
        monitor=score, min_delta=0.001, patience=30, verbose=False, mode=mode
    )
    model_checkpoint = ModelCheckpoint(
        monitor=score,
        dirpath=f"model_checkpoints/{model_name}",
        filename=f"{model_name}",
        save_top_k=1,
        save_last=True,
        mode=mode,
    )

    if conf["use_mlflow"]:
        logger.log_hyperparams(
            {
                "gradient_clip_val": conf["gradient_clip_val"],
                "model_name": conf["model_name"],
                "batch_size": conf["batch_size"],
                "epochs": conf["epochs"],
                "seed": conf["seed"],
                "prob_unk_token": conf["prob_unk_token"],
            }
        )
    trainer = Trainer(
        accelerator="cpu" if conf["n_gpus"] == 0 else "auto",
        devices=conf["n_gpus"] if conf["n_gpus"] > 1 else 1,
        strategy=DDPStrategy(find_unused_parameters=True) if conf["n_gpus"] > 1 else "auto",
        num_nodes=1,
        logger=logger,
        callbacks=[early_stop_callback, model_checkpoint],
        max_epochs=conf["epochs"],
        gradient_clip_val=conf["gradient_clip_val"],
    )

    log.info("Starting training")
    if ds_type == "classification":
        model.switch_to_classification()
    else:
        model.switch_to_generate()
    trainer.fit(model, data_module)
    best_val_score = model.best_val_score
    print(f"Best val loss: {best_val_score}")

    # Get the path to the best checkpoint
    best_model_path = model_checkpoint.best_model_path
    log.info(f"Best model path: {best_model_path}")
    # Load the best checkpoint
    best_model = TSVQTransformer.load_from_checkpoint(best_model_path)

    if ds_type == "classification":
        best_model.switch_to_classification()
    else:
        best_model.switch_to_generate()

    log.info("Training finished")
    trainer.test(best_model, dataloaders=data_module.test_dataloader())


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=check_dataset_name, default="ECG")
    args.add_argument("--model-name", type=str, default="TSVQ_Transformer")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--n-gpus", type=int, default=1)
    args.add_argument("--create-new-ds", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--d-model", type=int, default=256)
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--gradient-clip-val", type=float, default=0.8)
    args.add_argument("--prob-unk-token", type=float, default=0.0)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--n-resblocks", type=int, default=3)
    args.add_argument("--n-heads", type=int, default=8)
    args.add_argument("--embedding-classes", type=int, default=1024)
    args.add_argument("--epochs", type=int, default=30)
    args.add_argument("--att-dropout", type=float, default=0.0)
    args.add_argument("--res-dropout", type=float, default=0.1)
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--mlflow-experiment-name", type=str, default="xai-ts-classification")
    args.add_argument("--hyperparams-search-str", type=str, default="NoHyperparamSearch")
    args.add_argument("--dataset-path", type=str, default="data")
    args.add_argument("--vq-hidden-dim", type=int, default=32)
    args.add_argument("--vq-n-resblocks", type=int, default=6)
    args.add_argument("--beta", type=float, default=0.1)
    args.add_argument("--theta", type=float, default=0.5)
    args.add_argument("--gamma", type=float, default=0.5)
    args.add_argument("--use-class-head-bias", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--use-early-stopping", action=argparse.BooleanOptionalAction, default=True)
    conf = vars(args.parse_args())
    log.info(conf)
    seed_everything(conf["seed"])

    torch.set_float32_matmul_precision("medium")
    
    data_path = Path(conf["dataset_path"])
    conf["num_classes"] = get_num_classes(conf["dataset"])
    conf["data_dim"] = get_data_dim(conf["dataset"])
    
    model = init_transformer_model(conf)

    if (data_path / conf["dataset"] / "val_idx.npy").exists():
        val_idx, test_idx = load_val_test_idx(data_path / conf["dataset"])
    else:
        raise ValueError("val_idx.npy or test_idx.npy not found")

    log.info("Loading Reconstruction data")
    data_module: DATASET_MODULES = get_dataset_module(conf["dataset"])(
        dataset_path=data_path / conf["dataset"],
        ds_type="reconstruction",
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        batch_size=conf["batch_size"],
    )

    train_transformer(
        model=model,
        data_module=data_module,
        conf=conf,
        ds_type="reconstruction",
    )

    # log.info("Loading Classification data")
    # data_module: DATASET_MODULES = get_dataset_module(conf["dataset"])(
    #     dataset_path=data_path / conf["dataset"],
    #     ds_type="classification",
    #     val_split_idx=val_idx,
    #     test_split_idx=test_idx,
    #     batch_size=conf["batch_size"],
    # )

    # model.switch_to_classification()
    # train_transformer(
    #     model=model,
    #     data_module=data_module,
    #     conf=conf,
    #     ds_type="classification",
    # )


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
