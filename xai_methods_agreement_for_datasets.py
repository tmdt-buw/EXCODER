import argparse
import logging
from lightning import seed_everything
import torch
import torch.nn.functional as F
from itertools import combinations
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
)
from xai_methods.utils import load_explanations, run_evaluation, save_results

def main(conf: dict[str, any]):
    """
    Evaluate agreement between different XAI methods for the same model and dataset.
    
    This function measures the consistency between different explanation methods
    by computing the cosine similarity between explanation vectors from different methods.
    Higher similarity indicates greater agreement between explanation methods.
    
    Args:
        conf: Configuration dictionary containing:
            - dataset_name: Name of the dataset to evaluate
            - model_type: Type of model to evaluate
            - seed: Random seed for reproducibility
            - data_path: Path to data directory
            
    Returns:
        dict: Dictionary containing evaluation results:
            - mean_cosine_similarity: Average cosine similarity across all pairs of XAI methods
            - conf: Copy of the configuration
            
    Raises:
        FileNotFoundError: If no explanations could be loaded
    """
    seed_everything(conf["seed"])

    dataset_name: DATASET_NAMES = conf["dataset_name"]
    model_type: MODEL_NAMES = conf["model_type"]
    logging.info(f"Running {dataset_name} with {model_type}")

    torch.set_float32_matmul_precision("medium")

    all_seed_explanations = []

    xai_methods = ["SM", "IG", "RISE", "LIME"]
    for xai_method in xai_methods:
        try:
            explanations = load_explanations(
                dataset_name, 
                model_type, 
                xai_method, 
                conf["seed"], 
                conf["data_path"]
            )
            # min-max normalization
            explanations = (explanations - explanations.min()) / (explanations.max() - explanations.min())
            all_seed_explanations.append(explanations)
        except Exception as e:
            logging.error(f"Error loading {xai_method} explanations: {e}")

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

def run_xai_method(conf: dict[str, any]):
    """
    Run XAI method agreement evaluation for multiple datasets and models.
    
    This function coordinates the evaluation of agreement between different XAI methods
    (SM, IG, RISE, LIME) across a grid of datasets and models. Agreement is measured
    using cosine similarity between pairs of explanation methods.
    
    Args:
        conf: Base configuration dictionary that will be updated with specific
             settings for each evaluation run
             
    Returns:
        None: Results are saved to a pickle file at data_path/XAI_Results/xai_methods_agreement_results.pkl
    """
    # Define the datasets and models to evaluate
    conf.update({
        "datasets": ["ECG", "CNC_Machining", "Welding"],
        "models": ["SAX_MLP"],
        "num_seeds": 5
    })
    
    # Run the evaluation using the utility function
    result_list = run_evaluation(conf, main)
    
    # Save the results
    save_results(
        result_list, 
        "xai_methods_agreement_results.pkl", 
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
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--use-small-subset", type=bool, default=False)
    conf = vars(args.parse_args())

    run_xai_method(conf)