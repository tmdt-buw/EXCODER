import os
import argparse
from pathlib import Path
import logging
from copy import deepcopy
from lightning import seed_everything
import torch
import pickle
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
    XAI_METHODS,
)
import torch.nn.functional as F
from itertools import combinations

def main(conf: dict[str, any]):
    seed_everything(conf["seed"])
    ########################################################################################
    # choose from the following options: "CNC_Machining" | "Welding" | "ECG" | "UEA"
    dataset_name: DATASET_NAMES = conf["dataset_name"]
    # choose from the following options: "DLinear" | "MLP" | "TimesNet"
    model_type: MODEL_NAMES = conf["model_type"]
    logging.info(f"Running {dataset_name} with {model_type}")
    ########################################################################################

    torch.set_float32_matmul_precision("medium")

    all_seed_explanations = []

    xai_methods = ["SM", "IG", "RISE", "LIME"]
    for xai_method in xai_methods:
        conf["xai_method"] = xai_method
        project_path = Path(os.path.abspath(""))
        explanation_path = (
            project_path
            / conf["data_path"]
            / "XAI_Results"
            / dataset_name
            / model_type
            / f"seed_{conf['seed']}"
            / f"{conf['xai_method']}_explanations.pkl"
        )
        try:
            if not explanation_path.exists():
                raise FileNotFoundError(f"Explanations not found: {explanation_path}")
            with open(explanation_path, "rb") as f:
                explanation_dict = pickle.load(f)
            explanations = torch.tensor(explanation_dict["explanations"])
            # reshape to (dataset_size, input_size*in_dim) to show amount of features
            explanations = explanations.reshape(explanations.shape[0], -1)
            # min-max normalization
            explanations = (explanations - explanations.min()) / (explanations.max() - explanations.min())
            all_seed_explanations.append(explanations)
        except Exception as e:
            logging.error(f"Error loading {conf['xai_method']} explanations: {e}")

    if len(all_seed_explanations) == 0:
        raise FileNotFoundError("No explanations found")

    logging.info(f"{len(all_seed_explanations)} explanations found with shape: {all_seed_explanations[0].shape}")

    cosine_similarities_all = []
    for exp1, exp2 in combinations(all_seed_explanations, 2):
        cosine_similarities = F.cosine_similarity(exp1, exp2, dim=1)
        cosine_similarities_all.append(cosine_similarities)

    mean_cosine_similarity = torch.cat(cosine_similarities_all).mean().item()
    logging.info(f"Mean Cosine Similarity across all pairs: {mean_cosine_similarity}")

    dict_to_save = {
        "mean_cosine_similarity": mean_cosine_similarity,
        "conf": conf.copy(),
    }

    return dict_to_save
    # with open(output_path, "wb") as f:
    #     pickle.dump(dict_to_save, f)

def run_xai_method(conf: dict[str, any]):
    result_list = []
    models = ["SAX_MLP"]
    xai_methods = ["SM", "IG", "RISE", "LIME", "ATM"]
    datasets = ["ECG", "CNC_Machining", "Welding"]
    for dataset in datasets:
        conf["dataset_name"] = dataset
        for model in models:
            conf["model_type"] = model
            for seed in range(5):
                conf["seed"] = seed
                # for xai_method in xai_methods:
                #     conf["xai_method"] = xai_method   # xai_method iteration happens inside main
                try:
                    dict_to_save = deepcopy(main(conf))
                    result_list.append(dict_to_save)
                except Exception as e:
                    logging.error(f"Error running XAI Methods Agreement on {dataset} with {model}: {e}")
    
    output_path = Path(conf["data_path"]) / "XAI_Results" / "xai_methods_agreement_results.pkl"
    logging.info(f"Saving results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(result_list, f)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = argparse.ArgumentParser()
    args.add_argument("--data-path", type=str, default="data")
    args.add_argument("--dataset-name", type=str, default="ECG")
    args.add_argument("--model-type", type=str, default="VQ-VAE_MLP")
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--use-small-subset", type=bool, default=False)
    conf = vars(args.parse_args())

    # logging.info(f"Configuration: {conf}")
    # try:
        # dict_to_save = main(conf)
    # except Exception as e:
        # logging.error(f"Error running XAI Methods Agreement: {e}")
    run_xai_method(conf)