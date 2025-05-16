import logging as log
import argparse
import torch
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import seed_everything
from model.transformer_decoder import MyTransformerDecoder

from lightning import Trainer
from pathlib import Path
from data_loader.data_module import SimpleDataModule

from utils import (
    get_logger,
    check_dataset_name,
    get_num_classes,
    get_data_dim,
    load_first_stage_model,
    get_laten_ds,    
    convert_to_sax,
    load_val_test_idx,
    DATASET_MODULES,
    get_dataset_module,
)


def init_transformer_model(conf: dict[str, any]):
    log.info("Initializing Transformer model with the following configuration:")
    n_cycles = conf["n_cycles"]


    if conf["use_sax"]:
        in_seq_len = 8 * conf["data_dim"] + 1
        num_embeddings = conf["codebook_size"] + 3
    else:
        first_stage_model_config = load_first_stage_model(conf["path_vq_vae"])
        # Retrieve parameters from conf
        num_embeddings = first_stage_model_config.num_embeddings
     
        enc_out_len = first_stage_model_config.enc_out_len

        # Compute dependent variables
        in_seq_len = (enc_out_len * n_cycles) + 1
        num_embeddings = num_embeddings + 3
    log.info(
        f"Input sequence length: {in_seq_len} | Number of classification classes: {conf['num_classes']} | Number of generation classes: {num_embeddings}"
    )

    # Initialize the Transformer model
    transformer = MyTransformerDecoder(
        dataset_name=conf["dataset"],
        d_model=conf["d_model"],
        embedding_classes=num_embeddings,
        seq_len=in_seq_len,
        n_blocks=conf["n_resblocks"],
        n_head=conf["n_heads"],
        n_classes=conf["num_classes"],
        res_dropout=conf["res_dropout"],
        att_dropout=conf["att_dropout"],
        learning_rate=conf["learning_rate"],
        class_h_bias=conf["use_class_head_bias"],
    )

    return transformer


def get_new_trainer(
    epochs,
    logger,
    grad_clipping: float = 0.8,
    n_gpus=1,
    callbacks: list = [],
    use_cpu_only: bool = False,
):

    return Trainer(
        devices=n_gpus,
        accelerator="cpu" if use_cpu_only else "auto",
        num_nodes=1,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clipping,
        strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        accumulate_grad_batches=5,
    )


def classification_finetuning(
    model,
    class_task_data_module,
    classification_epoch,
    logger,
    gradient_clip: float = 0.8,
    n_gpus: int = 1,
    use_cpu_only: bool = False,
):
    callbacks = get_callbacks(
        is_classification=True, use_early_stopping=True, save_model=True
    )
    model.switch_to_classification()

    trainer = Trainer(
        devices=n_gpus,
        num_nodes=1,
        accelerator="cpu" if use_cpu_only else "auto",
        logger=logger,
        callbacks=callbacks,
        max_epochs=classification_epoch,
        gradient_clip_val=gradient_clip,
        strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        accumulate_grad_batches=5,
    )
    trainer.fit(model, class_task_data_module)
    best_val_score = model.best_val_score
    print(f"Best val loss: {best_val_score}")


def get_callbacks(
    is_classification: bool, use_early_stopping: bool, save_model: bool = False
):
    callbacks = []
    if is_classification:
        score = "val/f1_score"
        mode = "max"
    else:
        score = "val/loss"
        mode = "min"
    if save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath="model_checkpoints/VQ-VAE-transformer/",
            monitor=score,
            mode=mode,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    if use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=score, min_delta=0.001, patience=5, verbose=False, mode=mode
        )
        callbacks.append(early_stop_callback)
    return callbacks


def train_transformer(
    model, class_task_data_module, gen_task_data_module, conf: dict[str, any]
):
    use_cpu_only = conf["n_gpus"] == 0
    classification_epoch = conf["classification_epochs"]
    fine_tune_epochs = conf["finetune_epochs"]
    epoch_iter = conf["epoch_iter"]
    n_gpus = conf["n_gpus"]
    grad_clipping = conf["gradient_clip_val"]
    use_early_stopping = conf["use_early_stopping"]
    gen_epochs = conf["gen_epochs"]
    logger = get_logger(use_mlflow=conf["use_mlflow"], experiment_name=conf["mlflow_experiment_name"])

    if conf["use_mlflow"]:
        logger.log_hyperparams(
            {
                "gradient_clip_val": conf["gradient_clip_val"],
                "model_name": conf["model_name"],
                "batch_size": conf["batch_size"],
                "epoch_iter": conf["epoch_iter"],
                "classification_epochs": conf["classification_epochs"],
                "finetune_epochs": conf["finetune_epochs"],
                "gen_epochs": conf["gen_epochs"],
                "seed": conf["seed"],
                "path_vq_vae": str(conf["path_vq_vae"]),
                "embedding_model_name": conf["embedding_model_name"],
                "prob_unk_token": conf["prob_unk_token"],
            }
        )

    for epoch in range(epoch_iter):
        log.info("Genrerating stage")
        callbacks = get_callbacks(
            is_classification=False, use_early_stopping=use_early_stopping
        )
        trainer = get_new_trainer(
            epochs=gen_epochs,
            logger=logger,
            grad_clipping=grad_clipping,
            n_gpus=n_gpus,
            callbacks=callbacks,
            use_cpu_only=use_cpu_only,
        )
        model.switch_to_generate()
        trainer.fit(model, gen_task_data_module)

        if epoch == epoch_iter - 1:
            classification_finetuning(
                model=model,
                class_task_data_module=class_task_data_module,
                classification_epoch=fine_tune_epochs,
                gradient_clip=grad_clipping,
                logger=logger,
                n_gpus=n_gpus,
            )
        else:
            callbacks = get_callbacks(
                is_classification=True, use_early_stopping=use_early_stopping
            )
            trainer = get_new_trainer(
                epochs=classification_epoch,
                grad_clipping=grad_clipping,
                logger=logger,
                n_gpus=n_gpus,
                callbacks=callbacks,
                use_cpu_only=use_cpu_only,
            )
            log.info("Classification stage")
            model.switch_to_classification()
            trainer.fit(model, class_task_data_module)

    trainer = get_new_trainer(epochs=1, logger=logger, n_gpus=1, callbacks=callbacks)
    model.switch_to_classification()
    trainer.test(model, class_task_data_module)

    model.switch_to_generate()
    trainer.test(model, gen_task_data_module)

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=check_dataset_name, default="ECG")
    args.add_argument("--model-name", type=str, default="Transformer")
    args.add_argument("--embedding-model-name", type=str, default="SAX")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--n-gpus", type=int, default=1)
    args.add_argument("--create-new-ds", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--d-model", type=int, default=32)
    args.add_argument("--batch-size", type=int, default=1024)
    args.add_argument("--gradient-clip-val", type=float, default=0.8)
    args.add_argument("--prob-unk-token", type=float, default=0.0)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--n-resblocks", type=int, default=2)
    args.add_argument("--n-cycles", type=int, default=1)
    args.add_argument("--n-heads", type=int, default=8)
    args.add_argument("--epoch-iter", type=int, default=2)
    args.add_argument("--gen-epochs", type=int, default=1)
    args.add_argument("--classification-epochs", type=int, default=4)
    args.add_argument("--finetune-epochs", type=int, default=5)
    args.add_argument("--att-dropout", type=float, default=0.0)
    args.add_argument("--res-dropout", type=float, default=0.1)
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--mlflow-experiment-name", type=str, default="xai-ts-classification")
    args.add_argument("--hyperparams-search-str", type=str, default="NoHyperparamSearch")
    args.add_argument("--dataset-path", type=str, default="data")
    args.add_argument("--use-class-head-bias", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--use-early-stopping", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--use-sax", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--codebook-size", type=int, default=32)
    conf = vars(args.parse_args())
    log.info(conf)
    seed_everything(conf["seed"])

    conf["path_vq_vae"] = Path(f"model_checkpoints/best_models/{conf['embedding_model_name']}_{conf['dataset']}.ckpt")
    conf["num_classes"] = get_num_classes(conf["dataset"])
    conf["data_dim"] = get_data_dim(conf["dataset"])
    conf["model_name"] = conf["embedding_model_name"] + "_" + conf["model_name"]
    assert conf["embedding_model_name"] == "SAX" and conf["use_sax"], "Embedding model name must be SAX and use-sax must be True"

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
            task="autoregressive_classification",
        )

        data_module: DATASET_MODULES = get_dataset_module(conf["dataset"])(
            dataset_path=data_path / conf["dataset"],
            ds_type="reconstruction",
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
        )
        recon_train_ds, recon_val_ds, recon_test_ds = convert_to_sax(
            data_module=data_module,
            codebook_size=conf["codebook_size"],
            prob_unk_token=conf["prob_unk_token"],
            task="reconstruction",
        )
    else:
        (
            recon_train_ds,
            recon_val_ds,
            recon_test_ds,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = get_laten_ds(
            vq_vae_path=conf["path_vq_vae"],
            dataset_name=conf["dataset"],
            init_ds=conf["create_new_ds"],
            n_cycles=conf["n_cycles"],
            data_path=conf["dataset_path"],
            prob_unk_token=conf["prob_unk_token"],
        )

    gen_task_data_module = SimpleDataModule(
        train_ds=recon_train_ds,
        val_ds=recon_val_ds,
        test_ds=recon_test_ds,
        batch_size=conf["batch_size"],
    )

    class_task_data_module = SimpleDataModule(
        train_ds=class_train_ds,
        val_ds=class_val_ds,
        test_ds=class_test_ds,
        batch_size=conf["batch_size"],
    )

    torch.set_float32_matmul_precision("medium")
    model = init_transformer_model(conf)

    train_transformer(
        model=model,
        class_task_data_module=class_task_data_module,
        gen_task_data_module=gen_task_data_module,
        conf=conf,
    )



if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
