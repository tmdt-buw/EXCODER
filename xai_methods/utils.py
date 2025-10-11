import os
from pathlib import Path
import logging
import pickle
import torch
from copy import deepcopy


def load_explanations(dataset_name, model_type, xai_method, seed, data_path):
    """
    Load explanations from pickle files.
    
    Args:
        dataset_name: Name of the dataset
        model_type: Type of the model
        xai_method: XAI method name
        seed: Random seed
        data_path: Path to data directory
        
    Returns:
        torch.Tensor: Loaded explanations
    """
    project_path = Path(os.path.abspath(""))
    explanation_path = (
        project_path / data_path / "XAI_Results" / dataset_name / model_type
        / f"seed_{seed}" / f"{xai_method}_explanations.pkl"
    )
    if not explanation_path.exists():
        raise FileNotFoundError(f"Explanations not found: {explanation_path}")
    
    with open(explanation_path, "rb") as f:
        explanation_dict = pickle.load(f)
    explanations = torch.tensor(explanation_dict["explanations"])
    # Reshape to (dataset_size, input_size*in_dim)
    explanations = explanations.reshape(explanations.shape[0], -1)
    return explanations


def get_dataset_and_model(dataset_name, model_type, conf, get_classification_models_and_data, small_subset_size=1_000):
    """
    Load dataset and model.
    
    Args:
        dataset_name: Name of the dataset
        model_type: Type of the model
        conf: Configuration dictionary
        get_classification_models_and_data: Function to get models and data
        small_subset_size: Size of the small subset if used
    Returns:
        tuple: (model, dataset, target)
    """
    model, data_module = get_classification_models_and_data(
        dataset_name=dataset_name,
        model_type=model_type,
        batch_size=conf["batch_size"],
        data_path=conf["data_path"],
        seed=conf["seed"],
    )

    # Get the whole dataset as a single batch
    dataset_complete = data_module.test_dataloader().dataset
    dataset_size = len(dataset_complete)
    dataset = torch.stack([dataset_complete[i][0] for i in range(dataset_size)])
    target = torch.stack([dataset_complete[i][1] for i in range(dataset_size)])
    
    # In case you want to test a shorter dataset
    if conf.get("use_small_subset", False):
        dataset = dataset[:small_subset_size]
        target = target[:small_subset_size]
    
    return model, dataset, target


def get_train_dataset(data_module):
    """
    Get the training dataset.
    
    Args:
        data_module: Lightning data module
        
    Returns:
        tuple: (train_dataset, train_target)
    """
    train_dataset_complete = data_module.train_dataloader().dataset
    train_dataset_size = len(train_dataset_complete)
    train_dataset = torch.stack([train_dataset_complete[i][0] for i in range(train_dataset_size)])
    train_target = torch.stack([train_dataset_complete[i][1] for i in range(train_dataset_size)])
    return train_dataset, train_target


def get_model_predictions(model, dataset, batch_size=512, model_type="MLP"):
    """
    Get model predictions for a dataset.
    
    Args:
        model: The model
        dataset: Input dataset
        batch_size: Batch size for predictions
        model_type: Type of the model
        
    Returns:
        torch.Tensor: Model predictions
    """
    model_outputs = []
    dataset_size = dataset.shape[0]
    for j in range(0, dataset_size, batch_size):
        batch = dataset[j : j + batch_size].clone().to(model.device)
        if model_type in ["DVAE_Transformer", "VQ-VAE_Transformer"]:
            logits = model(batch.squeeze(-1), generate=False)
        else:
            logits = model(batch)
        model_outputs.append(logits.argmax(dim=1).cpu())
    return torch.cat(model_outputs)


def run_evaluation(conf, evaluation_function):
    """
    Run evaluation across datasets, models, XAI methods and seeds.
    
    Args:
        conf: Configuration dictionary containing lists of datasets, models, xai_methods
        evaluation_function: Function to evaluate each configuration
        
    Returns:
        list: List of evaluation results
    """
    result_list = []
    datasets = conf.get("datasets", [conf.get("dataset_name")])
    models = conf.get("models", [conf.get("model_type")])
    xai_methods = conf.get("xai_methods", [conf.get("xai_method")])
    num_seeds = conf.get("num_seeds", 5)
    
    for dataset_name in datasets:
        for model_type in models:
            for xai_method in xai_methods:
                for seed in range(num_seeds):
                    current_conf = deepcopy(conf)
                    current_conf.update({
                        "dataset_name": dataset_name,
                        "model_type": model_type,
                        "xai_method": xai_method,
                        "seed": seed
                    })
                    try:
                        result = evaluation_function(current_conf)
                        result_list.append(result)
                    except Exception as e:
                        logging.error(f"Error with {dataset_name}, {model_type}, {xai_method}, seed {seed}: {e}")
    return result_list


def save_results(result_list, output_file, data_path):
    """
    Save results to a pickle file.
    
    Args:
        result_list: List of results to save
        output_file: Output file name
        data_path: Path to data directory
    """
    output_path = Path(data_path) / "XAI_Results" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(result_list, f)


def get_accuracy_and_f1(model, dataset, target, f1_score, batch_size=512, model_type="MLP"):
    """
    Get accuracy and F1 score for a dataset.
    
    Args:
        model: The model
        dataset: Input dataset
        target: Target labels
        f1_score: F1 score metric
        batch_size: Batch size for predictions
        model_type: Type of the model
        
    Returns:
        tuple: (accuracy, f1_score)
    """
    model_outputs = get_model_predictions(model, dataset, batch_size, model_type)
    accuracy = (model_outputs == target.cpu()).float().mean()
    f1 = f1_score(model_outputs, target.cpu())
    return accuracy, f1
