import torch
from xai_methods.xai_blackbox_base import BlackBoxExplainer
import lime
import lime.lime_tabular
import numpy as np
from tqdm import tqdm
from pathlib import Path


def create_feature_strings(input_size: int = 17):
    strings = []
    for i in range(input_size):
        strings.append(f"Feature {i}")
    return strings


class Lime(BlackBoxExplainer):
    def __init__(
        self,
        model,
        dataset: np.ndarray,
        class_names: list,
        num_samples: int = 1000,
        use_latent_input: bool = True,
        model_type: str = "MLP",
        dataset_type: str = "CNC",
        discretize_continuous: bool = False,
        verbose: bool = False,
        conf=None,
    ):
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)

        self.num_samples = num_samples
        self.discretize_continuous = discretize_continuous
        self.verbose = verbose
        self.class_names = class_names
        self.input_size = model.input_size
        self.input_dim = model.in_dim
        self.dataset = dataset
        self.feature_names = create_feature_strings(model.input_size * model.in_dim)
        if use_latent_input:
            self.input_dim = 1
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.dataset.reshape(-1, self.input_size * self.input_dim),
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=range(self.input_size * self.input_dim),
            discretize_continuous=self.discretize_continuous,
            verbose=self.verbose,
        )

    def predict_fn(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Make sure that input_tensor is in the shape (Batch, Input_Size, Input_Dim)
        """
        if type(input_tensor) == np.ndarray:
            if self.model.hparams.get("use_latent_input", False):
                input_tensor = torch.tensor(input_tensor, dtype=torch.int64).to(
                    self.model.device
                )
            else:
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(
                    self.model.device
                )
        else:
            print("WARNING: Input was not a numpy array and therefore not typechecked.")
        input_tensor = input_tensor.view(-1, self.input_size, self.input_dim)
        probabilities_np = self.classify(input_tensor)
        return probabilities_np

    def explain(
        self,
        input_tensor: torch.Tensor | np.ndarray,
        return_exp_as_well: bool = False,
        save_to_pickle=False,
        save_path: str | Path = "lime_explanations.pkl",
    ) -> tuple:
        """
        Make sure the input_tensor gets turned in to an array
        """
        if type(input_tensor) == torch.Tensor:
            input_tensor = input_tensor.detach().numpy()
        predict_fn = lambda x: self.predict_fn(x)

        output = torch.empty(size=(len(input_tensor), self.input_size * self.input_dim))
        output = output.to(self.model.device)

        for i, instance in tqdm(
            enumerate(input_tensor),
            desc="Explaining instances",
            total=len(input_tensor),
        ):
            exp = self.explainer.explain_instance(
                instance.reshape(-1),
                predict_fn,
                num_features=self.input_size * self.input_dim,
                top_labels=1,
                num_samples=self.num_samples,
            )
            lime_exp = np.array(exp.local_exp[list(exp.local_exp.keys())[0]])
            lime_exp = lime_exp[lime_exp[:, 0].argsort()]
            lime_exp = torch.tensor(
                lime_exp[:, 1].reshape(self.input_size * self.input_dim)
            )
            output[i] = lime_exp

        if save_to_pickle:
            output = output.reshape(-1, self.input_size, self.input_dim)
            self.save(output, save_path)

        return output, exp if return_exp_as_well else None
