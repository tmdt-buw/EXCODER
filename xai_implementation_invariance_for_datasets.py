import argparse
import logging
from itertools import combinations
from lightning import seed_everything
import torch
import torch.nn.functional as F
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
    XAI_METHODS,
)
from xai_methods.utils import load_explanations, run_evaluation, save_results


def main(conf: dict[str, any]):
    """
    Evaluate implementation invariance of XAI methods across different random seeds.
    
    Implementation invariance measures how consistent an explanation method is when using
    different random seeds. This is calculated by computing the cosine similarity
    between explanations generated with different random seeds. Higher values indicate
    greater consistency and stability of the explanation method.
    
    Args:
        conf: Configuration dictionary containing:
            - dataset_name: Name of the dataset to evaluate
            - model_type: Type of model to evaluate
            - xai_method: XAI method to evaluate
            - data_path: Path to data directory
            
    Returns:
        dict: Dictionary containing evaluation results:
            - mean_cosine_similarity: Average cosine similarity across all explanation pairs
            - conf: Copy of the configuration
    """
    seed_everything(conf["seed"])
    
    dataset_name: DATASET_NAMES = conf["dataset_name"]
    model_type: MODEL_NAMES = conf["model_type"]
    xai_method: XAI_METHODS = conf["xai_method"]

    logging.info(f"Running {xai_method} on {dataset_name} with {model_type}")


    torch.set_float32_matmul_precision("medium")

    all_seed_explanations = []
    for seed in range(5):
        conf["seed"] = seed
        try:
            explanations = load_explanations(
                dataset_name, 
                model_type, 
                conf["xai_method"], 
                conf["seed"], 
                conf["data_path"]
            )
            # min-max normalization
            explanations = (explanations - explanations.min()) / (explanations.max() - explanations.min())
            all_seed_explanations.append(explanations)
        except FileNotFoundError as e:
            logging.error(f"Error loading explanations: {e}")

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

def run_xai_method(conf: dict[str, any]):
    """
    Run implementation invariance evaluation for multiple XAI methods, models, and datasets.
    
    This function coordinates the evaluation of implementation invariance across a grid of:
    - Datasets: ECG, CNC_Machining, Welding
    - Models: SAX_MLP
    - XAI methods: SM, IG, RISE, LIME, ATM
    
    Implementation invariance measures how consistent explanations are across different random seeds.
    
    Args:
        conf: Base configuration dictionary that will be updated with specific
             settings for each evaluation run
             
    Returns:
        None: Results are saved to a pickle file at data_path/XAI_Results/implementation_invariance_results.pkl
    """
    # Define the datasets and models to evaluate
    conf.update({
        "datasets": ["ECG", "CNC_Machining", "Welding"],
        "models": ["SAX_MLP"],
        "xai_methods": ["SM", "IG", "RISE", "LIME", "ATM"],
        "num_seeds": 5
    })
    
    # Run the evaluation using the utility function
    result_list = run_evaluation(conf, main)
    
    # Save the results
    save_results(
        result_list, 
        "implementation_invariance_results.pkl", 
        conf["data_path"]
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = argparse.ArgumentParser()
    args.add_argument("--data-path", type=str, default="data")
    args.add_argument("--dataset-name", type=str, default="ECG")
    args.add_argument("--model-type", type=str, default="VQ-VAE_MLP")
    args.add_argument("--xai-method", type=str, default="SM")
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--use-small-subset", type=bool, default=False)
    conf = vars(args.parse_args())

    logging.info(f"Configuration: {conf}")
    # main(conf)
    run_xai_method(conf)