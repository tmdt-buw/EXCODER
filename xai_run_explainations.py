import argparse
import logging
import time
import csv
from pathlib import Path
from datetime import datetime
from params import DATASET_NAMES, MODEL_NAMES, XAI_METHODS
from xai_explain_datasets import main as explain_datasets


CONF_MODELS: dict[MODEL_NAMES, dict[str, any]] = {
    "MLP": {
        "model_type": "MLP",
        "rise_mask_with_missing_category": False,
        "rise_classification_batch_size": 512,
        "rise_missing_category": 0,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "DLinear": {
        "model_type": "DLinear",
        "rise_mask_with_missing_category": False,
        "rise_classification_batch_size": 512,
        "rise_missing_category": 0,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "VQ-VAE_Transformer": {
        "model_type": "VQ-VAE_Transformer",
        "rise_mask_with_missing_category": True,
        "rise_missing_category": 0,
        "rise_classification_batch_size": 512,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "DVAE_Transformer": {
        "model_type": "DVAE_Transformer",
        "rise_mask_with_missing_category": True,
        "rise_missing_category": 0,
        "rise_classification_batch_size": 512,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "VQ-VAE_MLP": {
        "model_type": "VQ-VAE_MLP",
        "rise_mask_with_missing_category": True,
        "rise_missing_category": 0,
        "rise_classification_batch_size": 512,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "DVAE_MLP": {
        "model_type": "DVAE_MLP",
        "rise_mask_with_missing_category": True,
        "rise_missing_category": 0,
        "rise_classification_batch_size": 512,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "SAX_MLP": {
        "model_type": "SAX_MLP",
        "rise_mask_with_missing_category": True,
        "rise_missing_category": 0,
        "rise_classification_batch_size": 512,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 512,
    },
    "TS_Transformer": {
        "model_type": "TS_Transformer",
        "rise_mask_with_missing_category": False,
        "rise_classification_batch_size": 512,
        "rise_missing_category": 0,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 256,
    },
    "TimesNet": {
        "model_type": "TimesNet",
        "rise_mask_with_missing_category": False,
        "rise_classification_batch_size": 512,
        "rise_missing_category": 0,
        "rise_smooth_edges": False,
        "rise_sorted_codebook_distances": False,
        "rise_num_masks_per_instance": 1000,
        "rise_n_masked_percentage": 0.1,
        "rise_min_masking_value": 0,
        "lime_num_samples": 1000,
        "batch_size": 128,
    }
}


def get_standard_conf(
    model_type: MODEL_NAMES,
    dataset_name: DATASET_NAMES,
    xai_method: XAI_METHODS,
    use_small_subset: bool = False,
    data_path: str = "data",
):
    """
    Create a standard configuration dictionary for a specific model, dataset, and XAI method.
    
    Args:
        model_type: Type of the model to use
        dataset_name: Name of the dataset to use
        xai_method: XAI method to apply
        use_small_subset: Whether to use a small subset of the data
        data_path: Path to the data directory
        
    Returns:
        dict: Configuration dictionary with all required parameters
    """
    conf = CONF_MODELS[model_type]
    conf["seed"] = 0
    conf["dataset_name"] = dataset_name
    conf["xai_method"] = xai_method
    conf["use_small_subset"] = use_small_subset
    conf["data_path"] = data_path
    return conf


def timed_explain_datasets(conf: dict, csv_path: str = "timing_results.csv"):
    """
    Wrapper function that times the explain_datasets method and logs results to CSV.
    
    Args:
        conf: Configuration dictionary for explain_datasets
        csv_path: Path to the CSV file where timing results will be logged
        
    Returns:
        None: Executes explain_datasets and logs timing to CSV
    """
    # Extract attributes for logging
    model_type = conf["model_type"]
    dataset_name = conf["dataset_name"]
    xai_method = conf["xai_method"]
    seed = conf["seed"]
    
    # Record start time
    start_time = time.time()
    
    # Run the explain_datasets function
    try:
        explain_datasets(conf)
        success = True
        error_message = ""
    except Exception as e:
        success = False
        error_message = str(e)
        logging.error(f"Error running explanation: {e}")
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Prepare CSV row
    csv_row = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "dataset_name": dataset_name,
        "xai_method": xai_method,
        "seed": seed,
        "duration_seconds": duration,
        "success": success,
        "error_message": error_message
    }
    
    # Check if CSV file exists to determine if we need to write headers
    csv_file = Path(csv_path)
    file_exists = csv_file.exists()
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as f:
        fieldnames = ["timestamp", "model_type", "dataset_name", "xai_method", 
                      "seed", "duration_seconds", "success", "error_message"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(csv_row)
    
    logging.info(
        f"Completed {xai_method} for {dataset_name} with {model_type} "
        f"(seed {seed}) in {duration:.2f} seconds"
    )
    
    if not success:
        raise Exception(error_message)


def run_explainations(use_small_subset: bool = False, data_path: str = "data", csv_path: str = "timing_results.csv"):
    """
    Run explanations for various combinations of models, datasets, and XAI methods.
    
    This function runs a grid search over specified models, datasets, XAI methods, and seeds,
    generating explanations for each combination. For each combination, it creates a
    configuration and calls the explain_datasets function with timing.
    
    Args:
        use_small_subset: Whether to use a small subset of the data for testing
        data_path: Path to the data directory
        csv_path: Path to the CSV file where timing results will be logged
        
    Returns:
        None: Results are saved to disk as pickle files and timing is logged to CSV
    """
    dataset_names: list[DATASET_NAMES] = ["CNC_Machining", "Welding", "ECG"]
    model_types: list[MODEL_NAMES] = ["SAX_MLP", "VQ-VAE_MLP", "DVAE_MLP", "VQ-VAE_Transformer", "DVAE_Transformer", "TS_Transformer"]
    xai_methods: list[XAI_METHODS] = ["IG","SM", "RISE", "LIME"] # "IG", "SM", "RISE", "ATM", "LIME", "ATF"
    seeds: list[int] = [0, 1, 2, 3, 4]

    for model_type in model_types:
        for dataset_name in dataset_names:
            for xai_method in xai_methods:
                conf = get_standard_conf(
                    model_type, dataset_name, xai_method, use_small_subset, data_path
                )
                for seed in seeds:
                    conf["seed"] = seed
                    logging.info(
                        f"Running explainations for {conf['dataset_name']} with {conf['model_type']} and {conf['xai_method']} and seed {conf['seed']}"
                    )
                    timed_explain_datasets(conf, csv_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--use-small-subset", type=bool, default=True)
    parser.add_argument("--csv-path", type=str, default="timing_results.csv", 
                        help="Path to CSV file for logging timing results")
    args = parser.parse_args()
    use_small_subset = args.use_small_subset
    data_path = args.data_path
    csv_path = args.csv_path
    run_explainations(use_small_subset, data_path, csv_path)
