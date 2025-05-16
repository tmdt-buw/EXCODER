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
from utils import get_classification_models_and_data
from tqdm import tqdm
from torchmetrics import F1Score
from sklearn.metrics import auc


def return_predicted_class(model, model_type, input):
    if model_type in ["DVAE_Transformer", "VQ-VAE_Transformer"]:
        logits = model(input.squeeze(-1), generate=False)
    else:
        logits = model(input)
    return logits.argmax(dim=1)


def get_accuracy_and_f1_for_dataset(
    model,
    dataset,
    target,
    f1_score,
    classification_batch_size: int = 512,
    model_type: str = "MLP",
):
    # get list of predicted classes for each batch
    model_outputs = []
    dataset_size = dataset.shape[0]
    for j in range(0, dataset_size, classification_batch_size):
        batch = dataset[j : j + classification_batch_size].clone().to(model.device)
        output = return_predicted_class(model, model_type, batch)
        model_outputs.append(output.cpu())
    model_outputs = torch.cat(model_outputs)
    accuracy = (model_outputs == target).float().mean()
    f1 = f1_score(model_outputs, target)
    return accuracy, f1


def get_most_important_features_in_order(explanations: torch.Tensor):
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

        accuracy_while_perturbing[step], f1_while_perturbing[step] = (
            get_accuracy_and_f1_for_dataset(
                model,
                dataset_perturbed,
                target,
                f1_score,
                classification_batch_size,
                model_type,
            )
        )
    return dataset_perturbed, accuracy_while_perturbing, f1_while_perturbing


def get_dataset_and_model_and_explanations(
    dataset_name: DATASET_NAMES, model_type: MODEL_NAMES, conf: dict[str, any]
):
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
    logging.info(f"Dataset shape: {dataset.shape}")

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
        / f"{xai_method}_perturbations.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output path: {output_path}")

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

    unperturbed_accuracy, unperturbed_f1 = get_accuracy_and_f1_for_dataset(
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
    print(dict_to_save)
    return dict_to_save
    # with open(output_path, "wb") as f:
    #     pickle.dump(dict_to_save, f)


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
    output_path = Path(conf["data_path"]) / "XAI_Results" / "perturbation_results.pkl"
    result_list = load_perturbation_results(output_path)
    models = ["SAX_MLP"]
    # models = [
    #     "MLP",
    #     "DLinear",
    #     "VQ-VAE_Transformer",
    #     "VQ-VAE_MLP",
    #     "DVAE_Transformer",
    #     "DVAE_MLP",
    #     "TimesNet",
    #     "TS_Transformer",
    # ]
    xai_methods = ["SM", "IG", "RISE", "LIME", "ATM", "ATF", "RND"]
    datasets = ["ECG", "CNC_Machining", "Welding"]
    for dataset in datasets:
        conf["dataset_name"] = dataset
        for model in models:
            conf["model_type"] = model
            if model.startswith("DVAE") or model.startswith("VQ-VAE") or model.startswith("SAX"):
                conf["step_size"] = 1
            else:
                conf["step_size"] = 25
            for xai_method in xai_methods:
                conf["xai_method"] = xai_method
                for seed in range(5):
                    conf["seed"] = seed
                    if check_if_results_exist(result_list, conf):
                        logging.info(f"Results already exist for {xai_method} on {dataset} with {model} and seed {seed}")
                        continue
                    elif xai_method == "ATF" or xai_method == "ATM" and not model.endswith("_Transformer"):
                        logging.info(f"Skipping {xai_method} on {dataset} with {model} because it is not a transformer model")
                        continue
                    # try:
                    dict_to_save = deepcopy(main(conf))
                    result_list.append(dict_to_save)
                    # except Exception as e:
                    #     logging.error(
                    #         f"Error running {xai_method} on {dataset} with {model}: {e}"
                    #     )

    print(result_list)
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
    # main(conf)
    run_xai_method(conf)
