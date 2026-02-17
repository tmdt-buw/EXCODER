import logging as log
import argparse
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from data_loader.data_module import SimpleDataModule
from model.mlp import MLP
from lightning import Trainer
from utils import (
    get_logger,
    check_dataset_name,
    get_model_hash,
    get_num_classes,
    check_model_name,
    get_data_dim,
    load_first_stage_model,
    get_laten_ds,
    convert_to_sax,
    load_val_test_idx,
    DATASET_MODULES,
    get_dataset_module,
)


def init_model(conf: dict[str, any]):
    log.info("Initializing Latent Space model with the following configuration:")
    log.info(conf)

    if conf["use_sax"]:
        model_name = conf["model_name"]
        num_embeddings = conf["codebook_size"]
        in_seq_len = 8 * conf["data_dim"] + 1
    else:
        first_stage_model_config = load_first_stage_model(conf["path_vq_vae"])
        
        # Retrieve parameters from conf
        num_embeddings = first_stage_model_config.num_embeddings

        enc_out_len = first_stage_model_config.enc_out_len

        # Compute dependent variables
        in_seq_len = enc_out_len + 1
        model_name = conf["model_name"].split("_")[-1]
    log.info(f"Model name: {model_name}")
    if model_name == "MLP" or model_name == "SAX_MLP":
        model = MLP(
            input_size=in_seq_len,
            num_class=conf["num_classes"],
            in_dim=conf["data_dim"],
            n_hidden_layers=conf["n_hidden_layers"],
            d_model=conf["d_model"],
            dropout_p=conf["dropout_p"],
            learning_rate=conf["learning_rate"],
            use_layer_norm=conf["use_layer_norm"],
            use_latent_input=True,
            num_latent_tokens=num_embeddings + 3,
            hyperparams_search_str=conf["hyperparams_search_str"],
            dataset_name=conf["dataset"],
            hash_model=get_model_hash(conf),
        )
    else:
        raise ValueError(f"Model {conf['model_name']} not found")

    return model


def train_model(model, data_module: SimpleDataModule, conf: dict[str, any]):
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
    mlflow_logger = get_logger(
        use_mlflow=conf["use_mlflow"], experiment_name=conf["mlflow_experiment_name"]
    )

    # Log the gradient_clip_val
    if conf["use_mlflow"]:
        mlflow_logger.log_hyperparams(
            {
                "gradient_clip_val": conf["gradient_clip_val"],
                "prob-unk-token": conf["prob_unk_token"],
                "model_name": conf["model_name"],
                "batch_size": conf["batch_size"],
                "max_epochs": conf["epochs"],
                "seed": conf["seed"],
                "use_sax": conf["use_sax"],
                "codebook_size": conf["codebook_size"],
            }
        )

    model = init_model(conf)

    trainer = Trainer(
        accelerator="cpu" if conf["n_gpus"] == 0 else "auto",
        devices=conf["n_gpus"] if conf["n_gpus"] > 1 else 1,
        strategy=(
            DDPStrategy(find_unused_parameters=True) if conf["n_gpus"] > 1 else "auto"
        ),
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
    args.add_argument("--model-name", type=check_model_name, default="MLP")
    args.add_argument("--dataset", type=check_dataset_name, default="Welding")
    args.add_argument("--embedding-model-name", type=str, default="VQ-VAE")
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--n-resblocks", type=int, default=4)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--gradient-clip-val", type=float, default=1.0)
    args.add_argument("--dropout-p", type=float, default=0.1)
    args.add_argument("--hyperparams-search-str", type=str, default="NoHyperparamSearch")
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
    args.add_argument("--n-hidden-layers", type=int, default=4)
    args.add_argument("--input-size", type=int, default=200)
    args.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--mlflow-experiment-name", type=str, default="xai-ts-classification")
    args.add_argument("--prob-unk-token", type=float, default=0.0)
    args.add_argument("--create-new-ds", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--n-gpus", type=int, default=1)
    args.add_argument("--use-sax", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--codebook-size", type=int, default=32)
    conf = vars(args.parse_args())
    log.info(conf)
    pl.seed_everything(conf["seed"])

    conf["path_vq_vae"] = Path(f"model_checkpoints/best_models/{conf['embedding_model_name']}_{conf['dataset']}.ckpt")
    conf["num_classes"] = get_num_classes(conf["dataset"])
    conf["data_dim"] = get_data_dim(conf["dataset"])
    conf["model_name"] = conf["embedding_model_name"] + "_" + conf["model_name"]
    assert conf["embedding_model_name"] == "SAX" and conf["use_sax"], "Embedding model name must be SAX and use-sax must be True"
    torch.set_float32_matmul_precision("medium")
    data_path = Path(conf["dataset_path"])
    if conf["use_sax"]:
        val_idx, test_idx = load_val_test_idx(data_path / conf["dataset"])
        data_module: DATASET_MODULES = get_dataset_module(conf["dataset"])(
            dataset_path=data_path / conf["dataset"],
            ds_type="classification",
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
        )

        class_train_ds, class_val_ds, class_test_ds = convert_to_sax(
            data_module=data_module,
            codebook_size=conf["codebook_size"],
            prob_unk_token=conf["prob_unk_token"],
            task="classification",
        )


    else:
        (
            _,
            _,
            _,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = get_laten_ds(
            vq_vae_path=conf["path_vq_vae"],
            dataset_name=conf["dataset"],
            init_ds=conf["create_new_ds"],
            n_cycles=1,
            data_path=conf["dataset_path"],
            prob_unk_token=conf["prob_unk_token"],
            seq_prediction_task=False,
        )

    

    class_task_data_module = SimpleDataModule(
        train_ds=class_train_ds,
        val_ds=class_val_ds,
        test_ds=class_test_ds,
        batch_size=conf["batch_size"],
    )
    
    model = init_model(conf)

    train_model(
        model=model,
        data_module=class_task_data_module,
        conf=conf,
    )


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
