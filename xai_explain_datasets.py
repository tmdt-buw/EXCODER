import os
import argparse
from pathlib import Path
import logging
from pathlib import Path
from lightning import seed_everything
import numpy as np

import torch
from xai_methods.saliency_maps import Saliency_Maps
from xai_methods.lime import Lime
from xai_methods.rise import Rise
from xai_methods.shap import SHAP
from xai_methods.integrated_gradient import IntegratedGradients
from xai_methods.deeplift import DeepLift
from xai_methods.attention_map import AttentionMap
from utils import get_classification_models_and_data
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
    XAI_METHODS,
)


def get_dataset_and_model(
    dataset_name: DATASET_NAMES, model_type: MODEL_NAMES, conf: dict[str, any]
):
    model, data_module = get_classification_models_and_data(
        dataset_name=dataset_name,
        model_type=model_type,
        batch_size=conf["batch_size"],
        data_path=conf["data_path"],
        seed=conf["seed"],
    )

    # get the whole dataset as a single batch as tensor in shape (dataset_size, input_size, in_dim)
    dataset_complete = data_module.test_dataloader().dataset
    dataset_size = len(dataset_complete)
    dataset = torch.stack([dataset_complete[i][0] for i in range(dataset_size)])
    target = torch.stack([dataset_complete[i][1] for i in range(dataset_size)])
    # dataset = dataset.reshape(dataset_size, -1, model.in_dim)
    logging.info(f"Dataset shape: {dataset.shape}")

    # in case you want to test a shorter dataset
    if conf["use_small_subset"]:
        dataset = dataset[:200]
        target = target[:200]
    return model, dataset, target


def main(conf: dict[str, any]):
    seed_everything(conf["seed"])
    ########################################################################################
    # choose from the following options: "CNC_Machining" | "Welding" | "ECG" | "UEA"
    dataset_name: DATASET_NAMES = conf["dataset_name"]
    # choose from the following options: "DLinear" | "MLP" | "TimesNet"
    model_type: MODEL_NAMES = conf["model_type"]
    # choose from the following options: "LIME" | "RISE" | "SM"
    xai_method: XAI_METHODS = conf["xai_method"]

    logging.info(f"Running {xai_method} on {dataset_name} with {model_type}")

    ########################################################################################

    output_path = (
        Path(conf["data_path"])
        / "XAI_Results"
        / dataset_name
        / model_type
        / f"seed_{conf['seed']}"
        / f"{xai_method}_explanations.pkl"
    )
    if output_path.exists():
        logging.info(f"Output path already exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output path: {output_path}")

    torch.set_float32_matmul_precision("medium")

    model, dataset, target = get_dataset_and_model(dataset_name, model_type, conf)

    if model_type == "VQ-VAE_Transformer" or model_type == "DVAE_Transformer":
        model.hparams["use_latent_input"] = True
        model.input_size = (
            model.input_size if hasattr(model, "input_size") else model.seq_len
        )
        model.in_dim = model.in_dim if hasattr(model, "in_dim") else 1
        model.num_classes = (
            model.num_classes if hasattr(model, "num_classes") else model.n_classes
        )

        conf["rise_missing_category"] = model.embedding_classes - 1
        if xai_method == "RISE":
            assert (
                conf["rise_mask_with_missing_category"]
                and not conf["rise_smooth_edges"]
                and conf["rise_min_masking_value"] == 0
            ), f"RISE must mask with missing category when using latent input, {conf['rise_mask_with_missing_category']=}, {conf['rise_smooth_edges']=}, {conf['rise_min_masking_value']=}"
    elif model_type.endswith("_MLP"):
        conf["rise_missing_category"] = model.num_latent_tokens - 1
        model.in_dim = 1
        if xai_method == "RISE":
            assert (
                conf["rise_mask_with_missing_category"]
                and not conf["rise_smooth_edges"]
                and conf["rise_min_masking_value"] == 0
            ), f"RISE must mask with missing category when using latent input, {conf['rise_mask_with_missing_category']=}, {conf['rise_smooth_edges']=}, {conf['rise_min_masking_value']=}"

    if xai_method == "SM":
        saliency_maps = Saliency_Maps(
            model=model,
            use_latent_input=model.hparams.get("use_latent_input", False),
            model_type=model_type,
            dataset_type=dataset_name,
            conf=conf,
        )
        sm_explanations = saliency_maps.explain(
            input_tensor=dataset,
            target=target,
            save_to_pickle=True,
            save_path=str(output_path),
        )
        logging.info(f"Saliency Maps explanations shape: {sm_explanations.shape}")
    elif xai_method == "LIME":
        if dataset_name == "ECG":
            lime_class_names = ["N", "S", "V", "F", "Q"]  # names of the classes
        else:
            lime_class_names = ["Bad", "Good"]

        # lime_dataset = dataset.reshape(-1, model.input_size * model.in_dim)
        dataset_np = dataset.cpu().numpy()

        logging.info(
            f"LIME dataset shape: {dataset_np.shape} dtype: {dataset_np.dtype}"
        )
        lime = Lime(
            model=model,
            dataset=dataset_np,
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
            dataset_np,
            save_to_pickle=True,
            save_path=str(output_path),
        )
        lime_explanations = lime_explanations.reshape(
            -1, model.input_size, model.in_dim
        )
        logging.info(f"LIME explanations shape: {lime_explanations.shape}")

    elif xai_method == "RISE":
        logging.info(
            f"RISE use_latent_input: {model.hparams.get('use_latent_input', False)}"
        )
        rise = Rise(
            model=model,
            min_masking_value=conf["rise_min_masking_value"],
            n_masked_percentage=conf["rise_n_masked_percentage"],
            num_masks_per_instance=conf["rise_num_masks_per_instance"],
            mask_with_missing_category=conf["rise_mask_with_missing_category"],
            missing_category=conf["rise_missing_category"],
            classification_batch_size=conf["rise_classification_batch_size"],
            smooth_edges=conf["rise_smooth_edges"],
            sorted_codebook_distances=conf["rise_sorted_codebook_distances"],
            use_latent_input=model.hparams.get("use_latent_input", False),
            model_type=model_type,
            dataset_type=dataset_name,
            conf=conf,
        )
        logging.info(f"Input tensor shape: {dataset.shape} dtype: {dataset.dtype}")
        rise_explanations = rise.explain(
            dataset,
            save_to_pickle=True,
            save_path=str(output_path),
        )
        logging.info(f"Dataset explanations shape: {rise_explanations.shape}")
    elif xai_method == "SHAP":
        shap = SHAP(
            model=model,
            model_type=model_type,
            dataset_type=dataset_name,
            use_latent_input=model.hparams.get("use_latent_input", False),
            background_data=dataset,
            conf=conf,
        )
        shap_explanations = shap.explain(
            dataset,
            save_to_pickle=True,
            save_path=str(output_path),
        )
        logging.info(f"Dataset explanations shape: {shap_explanations.shape}")
    elif xai_method == "IG":
        integrated_gradients = IntegratedGradients(
            model=model,
            baseline=torch.zeros_like(dataset),
            model_type=model_type,
            dataset_type=dataset_name,
            use_latent_input=model.hparams.get("use_latent_input", False),
            conf=conf,
            batch_size=conf["batch_size"],
        )
        integrated_gradients.explain(
            dataset,
            target=target,
            save_to_pickle=True,
            save_path=str(output_path),
        )
    elif xai_method == "DeepLIFT":
        deeplift = DeepLift(
            model=model,
            model_type=model_type,
            dataset_type=dataset_name,
            use_latent_input=model.hparams.get("use_latent_input", False),
            baseline=torch.zeros_like(dataset),
            conf=conf,
        )
        deeplift.explain(dataset, save_to_pickle=True, save_path=str(output_path))
    elif xai_method == "ATM" or xai_method == "ATF":
        attention_map = AttentionMap(
            model=model,
            model_type=model_type,
            dataset_type=dataset_name,
            use_latent_input=model.hparams.get("use_latent_input", False),
            conf=conf,
            attention_type=xai_method
        )
        attention_map.explain(dataset, save_to_pickle=True, save_path=str(output_path))
    else:
        raise ValueError(f"XAI method {xai_method} not supported")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = argparse.ArgumentParser()
    args.add_argument("--dataset-name", type=str, default="ECG")
    args.add_argument("--model-type", type=str, default="MLP")
    args.add_argument("--xai-method", type=str, default="SHAP")
    args.add_argument("--data-path", type=str, default="data")
    args.add_argument("--lime-num-samples", type=int, default=1000)
    args.add_argument("--rise-min-masking-value", type=int, default=0)
    args.add_argument("--rise-n-masked-percentage", type=float, default=0.1)
    args.add_argument("--rise-num-masks-per-instance", type=int, default=1000)
    args.add_argument("--rise-mask-with-missing-category", type=bool, default=True)
    args.add_argument("--rise-missing-category", type=int, default=0)
    args.add_argument("--rise-classification-batch-size", type=int, default=32)
    args.add_argument("--rise-smooth-edges", type=bool, default=False)
    args.add_argument("--rise-sorted-codebook-distances", type=bool, default=None)
    args.add_argument("--shap-algorithm", type=str, default="auto")
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--shap-link", type=str, default="identity")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--use-small-subset", type=bool, default=False)
    conf = vars(args.parse_args())
    logging.info(f"Configuration: {conf}")
    main(conf)
