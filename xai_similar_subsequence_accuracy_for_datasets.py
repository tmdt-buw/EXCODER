import os
import argparse
from pathlib import Path
import logging
from copy import deepcopy
from lightning import seed_everything
import torch
import pickle
from tqdm import tqdm
from params import (
    DATASET_NAMES,
    MODEL_NAMES,
    XAI_METHODS,
)
from utils import get_classification_models_and_data


def return_predicted_class(model, model_type, input):
    """
    Get predicted class labels from a model for the given input.
    
    Args:
        model: The model to use for prediction
        model_type: Type of the model (special handling for transformer-based models)
        input: Input tensor to classify
        
    Returns:
        torch.Tensor: Predicted class indices (argmax of model output)
    """
    if model_type in ["DVAE_Transformer", "VQ-VAE_Transformer"]:
        logits = model(input.squeeze(-1), generate=False)
    else:
        logits = model(input)
    return logits.argmax(dim=1)

def get_model_predictions(
    model,
    dataset,
    classification_batch_size: int = 512,
    model_type: str = "MLP",
):
    """
    Get model predictions for an entire dataset in batches.
    
    Args:
        model: The model to use for prediction
        dataset: Input dataset tensor
        classification_batch_size: Batch size for inference
        model_type: Type of the model
        
    Returns:
        torch.Tensor: Predicted class indices for the entire dataset
    """
    model_outputs = []
    dataset_size = dataset.shape[0]
    for j in range(0, dataset_size, classification_batch_size):
        batch = dataset[j : j + classification_batch_size]
        output = return_predicted_class(model, model_type, batch)
        model_outputs.append(output)
    model_outputs = torch.cat(model_outputs)
    return model_outputs

def get_most_relevant_subsequences_starting_index(
    explanations: torch.Tensor, subseq_len: int
):
    """
    Find the starting index of the most relevant subsequence for each instance.
    
    This function identifies the most important subsequence of length subseq_len
    in each instance according to the explanation values. It does this by sliding
    a window of size subseq_len over the explanation values and summing them.
    
    Args:
        explanations: Tensor of explanation values with shape (dataset_size, input_size*in_dim)
        subseq_len: Length of subsequence to consider
        
    Returns:
        torch.Tensor: Starting indices of the most relevant subsequence for each instance
    """
    # assert that the shape of the explanations is (dataset_size, input_size*in_dim)
    assert explanations.dim() == 2
    dataset_size, input_size = explanations.shape

    amount_of_subsequences = input_size - subseq_len + 1
    subsequence_relevances = torch.zeros(dataset_size, amount_of_subsequences)
    most_relevant_subsequences = torch.zeros(dataset_size)

    for feature_index in range(amount_of_subsequences):
        subsequence_relevances[:, feature_index] = explanations[:, feature_index : feature_index + subseq_len].sum(dim=1)
    
    for i in range(dataset_size):
        most_relevant_subsequences[i] = subsequence_relevances[i].argmax()

    return most_relevant_subsequences.int()

def get_subsequences_from_indices(
    dataset: torch.Tensor, starting_indices: torch.Tensor, subseq_len: int
):
    """
    Extract subsequences from a dataset at given starting indices.
    
    Args:
        dataset: Input dataset tensor
        starting_indices: Starting indices for subsequence extraction for each instance
        subseq_len: Length of subsequences to extract
        
    Returns:
        torch.Tensor: Extracted subsequences with shape (dataset_size, subseq_len)
    """
    # reshape dataset to (dataset_size, input_size*in_dim)
    reshaped_dataset = dataset.reshape(dataset.size(0), -1)

    indices = starting_indices.unsqueeze(1) + torch.arange(subseq_len)
    subsequences_dataset = reshaped_dataset[torch.arange(reshaped_dataset.size(0)).unsqueeze(1), indices]
    return subsequences_dataset

def get_match_matrix_for_instances_with_same_subsequence(
    dataset: torch.Tensor, train_dataset: torch.Tensor, subsequences_dataset: torch.Tensor, starting_indices: torch.Tensor, subseq_len: int
):
    """
    Create a binary match matrix indicating which training instances have the same subsequence.
    
    For each instance in the dataset, this function identifies all instances in the training
    dataset that have the exact same subsequence at the same position.
    
    Args:
        dataset: Input dataset tensor
        train_dataset: Training dataset tensor
        subsequences_dataset: Extracted subsequences from the input dataset
        starting_indices: Starting indices of subsequences for each instance
        subseq_len: Length of subsequences
        
    Returns:
        torch.Tensor: Binary match matrix with shape (dataset_size, train_dataset_size)
                     where 1 indicates matching subsequences
    """
    # reshape dataset to (dataset_size, input_size*in_dim)
    reshaped_dataset = dataset.reshape(dataset.size(0), -1)
    reshaped_train_dataset = train_dataset.reshape(train_dataset.size(0), -1)
    reshaped_dataset_size, seq_length = reshaped_dataset.size()
    reshaped_train_dataset_size, _ = reshaped_train_dataset.size()
    match_matrix = torch.zeros((reshaped_dataset_size, reshaped_train_dataset_size), dtype=torch.int32)

    for i in tqdm(range(reshaped_dataset_size), desc="Matching instances"):
        start_idx = starting_indices[i]
        subsequence = subsequences_dataset[i]
         
        all_subsequences_in_same_pos = reshaped_train_dataset[:, start_idx : start_idx + subseq_len]
        match_matrix[i] = torch.all(all_subsequences_in_same_pos == subsequence, dim=1).int()
    return match_matrix

def analyze_match_matrix(match_matrix, ground_truth_labels, predictions):
    """
    Analyze the match matrix to compute correctness percentages.
    
    For each test instance, this function:
    1. Identifies all training instances with matching subsequences
    2. Calculates the percentage of those matching instances that have the same
       predicted class as the test instance
    
    Args:
        match_matrix: Binary matrix indicating matching subsequences
        ground_truth_labels: Ground truth labels for the training dataset
        predictions: Model predictions for the test dataset
        
    Returns:
        tuple:
            - num_matches: Number of matching instances for each test instance
            - percentage_correct: Percentage of matching instances with the same
                                 predicted class for each test instance
    """
    num_instances = match_matrix.size(0)#
    size_of_comparison_set = match_matrix.size(1)
    num_matches = match_matrix.sum(dim=1)
    percentage_correct = torch.zeros(num_instances, dtype=torch.float32)
    for i in tqdm(range(num_instances), desc="Analyzing match matrix"):
        matching_indices = match_matrix[i].nonzero(as_tuple=True)[0]
        if len(matching_indices) > 0:
            matching_labels = ground_truth_labels[matching_indices]
            current_prediction = predictions[i]
            correct_matches = (matching_labels == current_prediction).sum().item()
            percentage_correct[i] = (correct_matches / len(matching_indices)) * 100
    return num_matches, percentage_correct

def clean_up_match_matrix_analysis_and_return_ssa(
    num_matches: torch.Tensor, percentage_correct: torch.Tensor, min_amount_similar_subseq: int, target: torch.Tensor
):
    """
    Calculate final Similar Subsequence Accuracy (SSA) metrics.
    
    This function computes several SSA metrics:
    1. Mean SSA across instances with sufficient matches
    2. Weighted SSA (weighted by number of matches)
    3. Class-balanced SSA (average of class-specific SSAs)
    
    Args:
        num_matches: Number of matching instances for each test instance
        percentage_correct: Percentage of matching instances with same predicted class
        min_amount_similar_subseq: Minimum number of matches required to include an instance
        target: Ground truth labels for test instances
        
    Returns:
        tuple:
            - num_matches_enough_neighbours: Number of matches for valid instances
            - ssa_mean: Mean SSA score
            - ssa_weighted_mean: Mean SSA weighted by number of matches
            - ssa_meaned_over_classes: Mean SSA across classes (class-balanced)
    """
    target = target.cpu()
    valid_indices = (num_matches >= min_amount_similar_subseq)
    num_matches_enough_neighbours = num_matches[valid_indices]
    percentage_correct_enough_neighbours = percentage_correct[valid_indices]
    
    # weigh by the amount of similar subsequences
    total_matches = num_matches.sum().item()
    percentage_correct_weighted = percentage_correct * num_matches.float() / total_matches

    ssa_mean = percentage_correct_enough_neighbours.mean().item()
    ssa_weighted_mean = percentage_correct_weighted.sum().item()

    percentages_for_each_class = []
    for i in range (len(target.unique())): 
        class_indices = (target == i)
        # class_percentage_correct = percentage_correct[class_indices].mean()
        class_percentage_correct = (percentage_correct[class_indices] * num_matches[class_indices].float() / num_matches[class_indices].sum()).sum().item()
        percentages_for_each_class.append(class_percentage_correct)
        logging.info(f"Percentage correct for class {i}: {class_percentage_correct}")
    percentages_for_each_class = torch.tensor(percentages_for_each_class)
    ssa_meaned_over_classes = percentages_for_each_class.mean().item()

    return num_matches_enough_neighbours, ssa_mean, ssa_weighted_mean, ssa_meaned_over_classes

def get_dataset_and_model_and_explanations(
    dataset_name: DATASET_NAMES, model_type: MODEL_NAMES, conf: dict[str, any]
):
    """
    Load model, datasets, and explanations for the SSA evaluation.
    
    This function:
    1. Loads the model and test dataset
    2. Loads the training dataset for finding similar subsequences
    3. Loads pre-computed explanations or generates random explanations for baseline
    
    Args:
        dataset_name: Name of the dataset to load
        model_type: Type of model to load
        conf: Configuration dictionary containing:
            - xai_method: XAI method used for explanations
            - batch_size: Batch size for model operations
            - data_path: Path to data directory
            - seed: Random seed
            - use_small_subset: Whether to use a small subset of data
            
    Returns:
        tuple:
            - model: The loaded model
            - dataset: Test dataset
            - target: Test dataset labels
            - explanations: Feature importance values from XAI method
            - train_dataset: Training dataset
            - train_target: Training dataset labels
            
    Raises:
        FileNotFoundError: If explanation file cannot be found
    """
    # get the explanations
    project_path = Path(os.path.abspath(""))

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
    dataset = dataset.to(model.device)
    target = target.to(model.device)
    logging.info(f"Dataset shape: {dataset.shape}")

    train_dataset_complete = data_module.train_dataloader().dataset
    train_dataset_size = len(train_dataset_complete)
    train_dataset = torch.stack([train_dataset_complete[i][0] for i in range(train_dataset_size)])
    train_target = torch.stack([train_dataset_complete[i][1] for i in range(train_dataset_size)])
    train_dataset = train_dataset.to(model.device)
    train_target = train_target.to(model.device)
    logging.info(f"Train dataset shape: {train_dataset.shape}")	

    if conf["xai_method"] == "RND":
        # get random explanations for baseline methods
        explanations = torch.rand_like(dataset, dtype=torch.float32)
    else:
        explanation_path = (
            project_path
            / conf["data_path"]
            / "XAI_Results"
            / dataset_name
            / model_type
            / f"seed_{conf['seed']}"
            / f"{conf['xai_method']}_explanations.pkl"
        )
        if not explanation_path.exists():
            raise FileNotFoundError(f"Explanations not found: {explanation_path}")
        
        with open(explanation_path, "rb") as f:
            explanation_dict = pickle.load(f)
        explanations = torch.tensor(explanation_dict["explanations"])

    # reshape to (dataset_size, input_size*in_dim) to show amount of features
    explanations = explanations.reshape(dataset_size, -1)
    logging.info(f"Explanations shape: {explanations.shape}")

    # in case you want to test a shorter dataset
    if conf["use_small_subset"]:
        dataset = dataset[:200]
        target = target[:200]
        explanations = explanations[:200]
    return model, dataset, target, explanations, train_dataset, train_target

def prepare_for_subsequence_iteration(conf: dict[str, any]):
    """
    Prepare data for the SSA evaluation by loading and preprocessing everything needed.
    
    This function:
    1. Sets the random seed
    2. Loads the model, datasets and explanations
    3. Gets model predictions for the test dataset
    
    Args:
        conf: Configuration dictionary with parameters for the evaluation
        
    Returns:
        tuple:
            - dataset: Test dataset
            - target: Test dataset labels
            - explanations: Feature importance values from XAI method
            - model_predictions: Model's predictions for the test dataset
            - train_dataset: Training dataset
            - train_target: Training dataset labels
    """
    seed_everything(conf["seed"])

    dataset_name: DATASET_NAMES = conf["dataset_name"]
    model_type: MODEL_NAMES = conf["model_type"]
    xai_method: XAI_METHODS = conf["xai_method"]

    logging.info(f"Running {xai_method} on {dataset_name} with {model_type}")


    torch.set_float32_matmul_precision("medium")

    model, dataset, target, explanations, train_dataset, train_target = get_dataset_and_model_and_explanations(
        dataset_name, model_type, conf
    )
    
    # for each instance: get the model prediction
    model_predictions = get_model_predictions(model, dataset, conf["batch_size"], conf["model_type"])
    logging.info("Got all model predictions for the dataset.")

    return dataset, target, explanations, model_predictions, train_dataset, train_target

def main(conf: dict[str, any], dataset, target, explanations, model_predictions, train_dataset, train_target):
    """
    Calculate Similar Subsequence Accuracy (SSA) for the given dataset and explanations.
    
    This function evaluates explanation methods by finding the most important subsequences
    according to the explanations, then checking if instances with the same subsequence
    in the same position tend to have the same class. Higher SSA scores indicate better
    explanations, as they identify subsequences that are truly predictive of the class.
    
    Args:
        conf: Configuration dictionary with parameters
        dataset: Input dataset tensor
        target: Ground truth labels
        explanations: Feature importance values from XAI method
        model_predictions: Model's predictions for the dataset
        train_dataset: Training dataset for finding instances with similar subsequences
        train_target: Training dataset labels
        
    Returns:
        dict: Dictionary containing evaluation results:
            - ssa_mean: Mean SSA score
            - ssa_weighted_mean: Mean SSA weighted by number of matches
            - ssa_meaned_over_classes: Mean SSA across classes
            - num_matches: Number of similar subsequence matches per instance
            - percentage_correct: Percentage of correct predictions per instance
            - instances_with_enough_neighbours: Count of instances with sufficient matches
            - conf: Copy of the configuration
    """
    # for each instance: get the index of the most relevant subsequence
    most_relevant_subsequences_starting_index = get_most_relevant_subsequences_starting_index(explanations=explanations, subseq_len=conf["subseq_len"])

    # for each instance: get the actual most relevant subsequence by querying the dataset with the most relevant subsequence index
    subsequences_dataset = get_subsequences_from_indices(dataset, most_relevant_subsequences_starting_index, conf["subseq_len"])

    # get a binary matrix where each row represents the instances and each column represents the instances with a 1 if they have the same subsequence in the same position
    match_matrix = get_match_matrix_for_instances_with_same_subsequence(dataset, train_dataset, subsequences_dataset, most_relevant_subsequences_starting_index, conf["subseq_len"])

    # for each instance: analyze the match matrix and return the number of matches and the percentage of instances that have the same ground truth label as predicted by the model for the respective instance
    num_matches, percentage_correct = analyze_match_matrix(match_matrix, train_target, model_predictions)

    # for each instance: clean up the analysis by removing instances with less than min_amount_similar_subseq or more than max_amount_similar_subseq and return the SSA, which is the average percentage of correct predictions for the instances with valid amount of similar subsequences
    num_matches_enough_neighbours, ssa_mean, ssa_weighted_mean, ssa_meaned_over_classes = clean_up_match_matrix_analysis_and_return_ssa(num_matches, percentage_correct, conf["min_amount_similar_subseq"], target)

    logging.info("Average amount of similar subsequences: {:.2f}".format(num_matches.float().mean().item()))
    logging.info(f"Instances with at least {conf['min_amount_similar_subseq']} \"neighbours\": {len(num_matches_enough_neighbours)}/{len(num_matches)}")
    logging.info("SSA Mean: {:.2f}".format(ssa_mean))
    logging.info("SSA Weighted Mean: {:.2f}".format(ssa_weighted_mean))
    logging.info("SSA Meaned over classes: {:.2f}".format(ssa_meaned_over_classes))

    dict_to_save = {
        "ssa_mean": ssa_mean,
        "ssa_weighted_mean": ssa_weighted_mean,
        "ssa_meaned_over_classes": ssa_meaned_over_classes,
        "num_matches": num_matches,
        "percentage_correct": percentage_correct,
        "instances_with_enough_neighbours": len(num_matches_enough_neighbours),
        "conf": conf.copy(),
    }
    return dict_to_save
    # with open(output_path, "wb") as f:
    #     pickle.dump(dict_to_save, f)

def run_xai_method(conf: dict[str, any]):
    """
    Run SSA evaluation for multiple datasets, models, XAI methods, and subsequence lengths.
    
    This function coordinates the evaluation process across a grid of parameters:
    - Datasets: ECG, CNC_Machining, Welding
    - Models: SAX_MLP
    - XAI methods: SM, IG, RISE, LIME, ATM, RND
    - Random seeds: 0-4
    - Subsequence lengths: 1-5
    
    For each configuration, it calculates the Similar Subsequence Accuracy (SSA)
    and saves all results to a pickle file.
    
    Args:
        conf: Base configuration dictionary that will be updated with specific
             settings for each evaluation run
             
    Returns:
        None: Results are saved to a pickle file at data_path/XAI_Results/ssa_results.pkl
    """
    result_list = []
    # models = ["VQ-VAE_Transformer", "VQ-VAE_MLP", "DVAE_Transformer", "DVAE_MLP"]
    models = ["SAX_MLP"]

    xai_methods = ["SM", "IG", "RISE", "LIME", "ATM", "RND"]
    datasets = ["ECG", "CNC_Machining", "Welding"]

    for dataset_name in datasets:
        conf["dataset_name"] = dataset_name
        for model in models:
            conf["model_type"] = model
            for xai_method in xai_methods:
                conf["xai_method"] = xai_method
                for seed in range(5):
                    conf["seed"] = seed
                    try:
                        dataset, target, explanations, model_predictions, train_dataset, train_target = prepare_for_subsequence_iteration(conf)
                        for subseq_len in [1, 2, 3, 4, 5]:
                            conf["subseq_len"] = subseq_len
                            print("Subsequence length set to ", subseq_len)
                            dict_to_save = deepcopy(main(conf, dataset, target, explanations, model_predictions, train_dataset, train_target))
                            result_list.append(dict_to_save)
                    except Exception as e:
                        logging.error(f"Error preparing {xai_method} on {dataset_name} with {model}: {e}")
    
    output_path = Path(conf["data_path"]) / "XAI_Results" / "ssa_results.pkl"
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
    args.add_argument("--xai-method", type=str, default="SM")
    args.add_argument("--subseq_len", type=int, default=1)
    args.add_argument("--min-amount-similar-subseq", type=int, default=20)
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--use-small-subset", type=bool, default=False)
    conf = vars(args.parse_args())
    logging.info(f"Configuration: {conf}")



    run_xai_method(conf)