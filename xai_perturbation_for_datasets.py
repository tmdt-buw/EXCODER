import argparse
from pathlib import Path
import logging
from copy import deepcopy
from lightning import seed_everything
import torch
import pickle
from tqdm import tqdm
from torchmetrics import F1Score
from sklearn.metrics import auc
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
    XAI_METHODS,
)
from utils import get_classification_models_and_data
from xai_methods.utils import (
    load_explanations, 
    get_dataset_and_model, 
    get_accuracy_and_f1
)



def get_most_important_features_in_order(explanations: torch.Tensor):
    """
    Sort features by their importance according to the XAI explanations.
    
    Args:
        explanations: Tensor containing feature importance values for each instance
                     in shape (dataset_size, input_size*input_dim)
                    
    Returns:
        torch.Tensor: Indices of features sorted by importance (most to least) for each instance
                     in shape (dataset_size, input_size*input_dim)
    """
    assert (
        len(explanations.shape) == 2
    ), "Explanations should be in shape (dataset_size, input_size*input_dim)"
    # sort the indexes of features by value for each instance
    max_indices = torch.argsort(explanations, dim=1, descending=True)
    return max_indices


def perturb(
    feature_order: torch.tensor,
    dataset: torch.tensor,
    target: torch.tensor,
    model,
    classification_batch_size: int = 512,
    step_size: int = 10,
    f1_score=None,
    model_type: str = "MLP",
    missing_category: int = 128,
):
    """
    Progressively perturb features in order of importance and measure performance degradation.
    
    This function iteratively perturbs the most important features in the dataset
    according to the provided order and measures how model performance decreases.
    For each step, it perturbs step_size additional features, replacing them
    with zeros or a missing category token.
    
    Args:
        feature_order: Indices of features sorted by importance for each instance
        dataset: Input dataset to perturb
        target: Ground truth labels 
        model: The model to evaluate
        classification_batch_size: Batch size for model inference
        step_size: Number of features to perturb in each step
        f1_score: F1 score metric instance
        model_type: Type of the model
        missing_category: Value to use for perturbation in latent models
        
    Returns:
        tuple: 
            - Perturbed dataset after all perturbation steps
            - Accuracy at each perturbation step
            - F1 score at each perturbation step
    """
    dataset_perturbed = dataset.clone()
    use_latent_input = model.hparams.get("use_latent_input", False)
    seq_len = feature_order.shape[1]
    step_n = int(seq_len / step_size)
    accuracy_while_perturbing = torch.zeros(step_n)
    f1_while_perturbing = torch.zeros(step_n)
    dataset_size = dataset.shape[0]
    classification_batch_size = classification_batch_size
    for step, i in tqdm(
        enumerate(range(0, seq_len, step_size)),
        desc=f"Perturbing Dataset in {step_n} steps, perturbing {step_size} features at a time",
        total=step_n,
    ):
        current_feature_order = feature_order[:, i : i + step_size]
        row_indices = (
            torch.arange(current_feature_order.size(0))
            .unsqueeze(1)
            .expand_as(current_feature_order)
        )

        # reshape dataset for perturbation in shape (dataset_size, input_size*in_dim)
        dataset_perturbed = dataset_perturbed.reshape(dataset_size, -1)

        if use_latent_input:
            dataset_perturbed[row_indices, current_feature_order] = missing_category
        else:
            dataset_perturbed[row_indices, current_feature_order] = 0

        # shape dataset back into (dataset_size, input_size, in_dim)
        if model_type == "VQ-VAE_MLP" or model_type == "DVAE_MLP" or model_type == "SAX_MLP":
            dataset_perturbed = dataset_perturbed.reshape(dataset_size, -1)
        else:
            dataset_perturbed = dataset_perturbed.reshape(
                dataset_size, -1, model.in_dim
            )

        accuracy_while_perturbing[step], f1_while_perturbing[step] = get_accuracy_and_f1(
            model,
            dataset_perturbed,
            target,
            f1_score,
            classification_batch_size,
            model_type,
        )
    return dataset_perturbed, accuracy_while_perturbing, f1_while_perturbing


def get_dataset_and_model_and_explanations(
    dataset_name: DATASET_NAMES, model_type: MODEL_NAMES, conf: dict[str, any]
):
    """
    Load the dataset, model, and XAI explanations.
    
    This function handles loading the model and dataset, and retrieves
    the pre-computed explanations for the specified XAI method. For baseline
    comparison, it can generate random explanations if requested.
    
    Args:
        dataset_name: Name of the dataset to load
        model_type: Type of model to load
        conf: Configuration dictionary containing:
            - xai_method: XAI method to use
            - seed: Random seed for reproducibility
            - data_path: Path to data directory
            - step_size: Number of features to perturb in each step
            - use_small_subset: Whether to use a small subset of data
            
    Returns:
        tuple:
            - model: The loaded model
            - dataset: The input dataset
            - target: Ground truth labels
            - explanations: Explanations from the XAI method
            
    Raises:
        FileNotFoundError: If explanation file cannot be found
        AssertionError: If step_size is invalid for the dataset
    """
    model, dataset, target = get_dataset_and_model(
        dataset_name, model_type, conf, get_classification_models_and_data
    )

    if conf["xai_method"] == "RND":
        # get random explanations for baseline methods
        explanations = torch.rand_like(dataset, dtype=torch.float32)
    else:
        try:
            explanations = load_explanations(
                dataset_name, 
                model_type, 
                conf["xai_method"], 
                conf["seed"], 
                conf["data_path"]
            )
        except FileNotFoundError as e:
            logging.error(f"Error loading explanations: {e}")
            raise

    # reshape to (dataset_size, input_size*in_dim) to show amount of features
    dataset_size = dataset.shape[0]
    explanations = explanations.reshape(dataset_size, -1)
    logging.info(f"Explanations shape: {explanations.shape}")

    assert (
        conf["step_size"] <= dataset.shape[1]
        and explanations.shape[1] % conf["step_size"] == 0
    ), f"Step Size for Pertubation tests must be smaller than sequence length an should be a divisor of the input size. Step Size: {conf['step_size']}, Sequence Length: {dataset.shape[1]}"

    # in case you want to test a shorter dataset
    if conf["use_small_subset"]:
        dataset = dataset[:200]
        target = target[:200]
        explanations = explanations[:200]
    return model, dataset, target, explanations


def main(conf: dict[str, any]):
    """
    Evaluate XAI methods by perturbing features in order of importance.
    
    This function evaluates the quality of explanations by progressively perturbing
    the most important features according to the explanation and measuring the 
    degradation in model performance. The results are quantified using the area 
    under the accuracy and F1 curves.
    
    Args:
        conf: Dictionary containing configuration parameters including:
            - dataset_name: Name of the dataset to use
            - model_type: Type of model to evaluate
            - xai_method: XAI method to evaluate
            - step_size: Number of features to perturb in each step
            - batch_size: Batch size for model inference
            - seed: Random seed for reproducibility
            - data_path: Path to data directory
            - use_small_subset: Whether to use a small subset of the data
            
    Returns:
        dict: Dictionary containing evaluation results:
            - complete_accuracy: Accuracy at each perturbation step
            - complete_f1: F1 score at each perturbation step
            - auc_score_accuracy: Area under the accuracy curve
            - auc_score_f1: Area under the F1 curve
            - percentage_perturbed_x_axis: Percentage of features perturbed
            - conf: Copy of the configuration
    """
    seed_everything(conf["seed"])
    dataset_name: DATASET_NAMES = conf["dataset_name"]
    model_type: MODEL_NAMES = conf["model_type"]
    xai_method: XAI_METHODS = conf["xai_method"]

    logging.info(f"Running {xai_method} on {dataset_name} with {model_type}")

    torch.set_float32_matmul_precision("medium")

    model, dataset, target, explanations = get_dataset_and_model_and_explanations(
        dataset_name, model_type, conf
    )
    feature_order = get_most_important_features_in_order(explanations)

    if model_type.endswith("_Transformer"):
        model.hparams["use_latent_input"] = True
        model.input_size = (
            model.input_size if hasattr(model, "input_size") else model.seq_len
        )
        model.in_dim = model.in_dim if hasattr(model, "in_dim") else 1
        model.num_classes = (
            model.num_classes if hasattr(model, "num_classes") else model.n_classes
        )
        conf["missing_category"] = model.embedding_classes - 1

    elif model_type.endswith("_MLP"):
        conf["missing_category"] = model.num_latent_tokens - 1
    else:
        conf["missing_category"] = 0

    f1_score = F1Score(
        task="multiclass", num_classes=model.num_classes, average="macro"
    )

    unperturbed_accuracy, unperturbed_f1 = get_accuracy_and_f1(
        model, dataset, target, f1_score, conf["batch_size"], model_type
    )
    logging.info(f"Unperturbed Accuracy: {unperturbed_accuracy}")
    logging.info(f"Unperturbed F1: {unperturbed_f1}")

    _, accuracy_while_perturbing, f1_while_perturbing = perturb(
        feature_order=feature_order,
        dataset=dataset,
        target=target,
        model=model,
        classification_batch_size=conf["batch_size"],
        step_size=conf["step_size"],
        f1_score=f1_score,
        model_type=model_type,
        missing_category=conf["missing_category"],
    )

    complete_accuracy = torch.cat(
        [unperturbed_accuracy.unsqueeze(0), accuracy_while_perturbing]
    )
    complete_f1 = torch.cat([unperturbed_f1.unsqueeze(0), f1_while_perturbing])
    if len(dataset.shape) == 3:
        step_n = int(dataset.shape[1] * dataset.shape[2] / conf["step_size"])
    else:
        step_n = int(dataset.shape[1] / conf["step_size"])
    percentage_perturbed_x_axis = torch.linspace(0, 1, step_n + 1)
    auc_score_accuracy = auc(percentage_perturbed_x_axis, complete_accuracy)
    auc_score_f1 = auc(percentage_perturbed_x_axis, complete_f1)

    logging.info(f"Area under the curve for accuracy: {auc_score_accuracy}")
    logging.info(f"Area under the curve for f1: {auc_score_f1}")

    dict_to_save = {
        "complete_accuracy": complete_accuracy.numpy(),
        "complete_f1": complete_f1.numpy(),
        "auc_score_accuracy": auc_score_accuracy,
        "auc_score_f1": auc_score_f1,
        "percentage_perturbed_x_axis": percentage_perturbed_x_axis.numpy(),
        "conf": conf.copy(),
    }
    return dict_to_save


def load_perturbation_results(path: Path):
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def check_if_results_exist(list_results: list[dict[str, any]], conf: dict[str, any]):
    for result in list_results:
        if result["conf"] == conf:
            return True
    return False


def run_xai_method(conf: dict[str, any]):
    """
    Run perturbation-based evaluations for multiple XAI methods, models, and datasets.
    
    This function coordinates the evaluation process by:
    1. Loading existing results if available
    2. Setting up a grid of datasets, models, XAI methods, and seeds
    3. Running the perturbation-based evaluation for each configuration
    4. Saving the results to disk
    
    The function skips configurations that already have results or are invalid
    (e.g., attention-based methods on non-transformer models).
    
    Args:
        conf: Base configuration dictionary that will be updated with specific
             settings for each evaluation run
             
    Returns:
        None: Results are saved to a pickle file at data_path/XAI_Results/perturbation_results.pkl
    """
    output_path = Path(conf["data_path"]) / "XAI_Results" / "perturbation_results.pkl"
    result_list = load_perturbation_results(output_path)

    # Define configurations for evaluation
    conf.update({
        "datasets": ["ECG", "CNC_Machining", "Welding"],
        "models": ["SAX_MLP"],
        "xai_methods": ["SM", "IG", "RISE", "LIME", "ATM", "ATF", "RND"],
    })
    
    updated_results = []
    
    for dataset_name in conf["datasets"]:
        current_conf = conf.copy()
        current_conf["dataset_name"] = dataset_name
        
        for model_type in conf["models"]:
            current_conf["model_type"] = model_type
            
            # Set step size based on model type
            if model_type.startswith("DVAE") or model_type.startswith("VQ-VAE") or model_type.startswith("SAX"):
                current_conf["step_size"] = 1
            else:
                current_conf["step_size"] = 25
                
            for xai_method in conf["xai_methods"]:
                current_conf["xai_method"] = xai_method
                
                for seed in range(5):
                    current_conf["seed"] = seed
                    
                    # Skip if result already exists
                    if check_if_results_exist(result_list, current_conf):
                        logging.info(f"Results already exist for {xai_method} on {dataset_name} with {model_type} and seed {seed}")
                        continue
                        
                    # Skip if attention-based methods on non-transformer models
                    elif (xai_method == "ATF" or xai_method == "ATM") and not model_type.endswith("_Transformer"):
                        logging.info(f"Skipping {xai_method} on {dataset_name} with {model_type} because it is not a transformer model")
                        continue
                        
                    try:
                        dict_to_save = deepcopy(main(current_conf))
                        updated_results.append(dict_to_save)
                    except Exception as e:
                        logging.error(f"Error running {xai_method} on {dataset_name} with {model_type}: {e}")
    
    # Add any new results to the existing ones
    result_list.extend(updated_results)
    
    # Save results
    with open(output_path, "wb") as f:
        pickle.dump(result_list, f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = argparse.ArgumentParser()
    args.add_argument("--data-path", type=str, default="data")
    args.add_argument("--dataset-name", type=str, default="ECG")
    args.add_argument("--model-type", type=str, default="MLP")
    args.add_argument("--xai-method", type=str, default="SM")
    args.add_argument("--step-size", type=int, default=25)
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--use-small-subset", type=bool, default=False)
    conf = vars(args.parse_args())
    logging.info(f"Configuration: {conf}")
    run_xai_method(conf)
