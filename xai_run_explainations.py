import argparse
import logging
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
    conf = CONF_MODELS[model_type]
    conf["seed"] = 0
    conf["dataset_name"] = dataset_name
    conf["xai_method"] = xai_method
    conf["use_small_subset"] = use_small_subset
    conf["data_path"] = data_path
    return conf


def run_explainations(use_small_subset: bool = False, data_path: str = "data"):

    dataset_names: list[DATASET_NAMES] = ["CNC_Machining", "Welding", "ECG"]
    model_types: list[MODEL_NAMES] = ["SAX_MLP"] # "VQ-VAE_MLP", "DVAE_MLP" "VQ-VAE_Transformer", "DVAE_Transformer", "TS_Transformer"
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
                    explain_datasets(conf)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--use-small-subset", type=bool, default=False)
    args = parser.parse_args()
    use_small_subset = args.use_small_subset
    data_path = args.data_path
    run_explainations(use_small_subset, data_path)
