from typing import Literal
import torch
from tqdm import tqdm
from xai_methods.xai_blackbox_base import BlackBoxExplainer


class AttentionMap(BlackBoxExplainer):
    def __init__(
        self,
        model,
        model_type: str,
        dataset_type: str,
        use_latent_input: bool,
        conf=None,
        classification_batch_size: int = 128,
        attention_type: Literal["ATM", "ATF"] = "ATM",
    ):
        super().__init__(model, model_type, dataset_type, use_latent_input, conf)

        # check if model has function get_attention_weights
        assert hasattr(
            model, "get_attention_weights"
        ), "Model does not have get_attention_weights function"
        self.input_size = self.model.input_size
        self.input_dim = self.model.in_dim
        self.num_class = self.model.num_classes
        self.classification_batch_size = classification_batch_size
        self.attention_type = attention_type
        
    @torch.no_grad()
    def explain(
        self,
        input_tensor: torch.Tensor,
        save_to_pickle=False,
        save_path="attention_map_explanations.pkl",
    ):
        output = torch.empty(size=(len(input_tensor), self.input_size * self.input_dim))
        output = output.to(self.model.device)

        # batch input tensor
        if self.model_type == "TS_Transformer":
            input_tensor = input_tensor.reshape(input_tensor.shape[0], self.input_size, self.input_dim)
        else:
            input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

        for i in tqdm(
            range(0, len(input_tensor), self.classification_batch_size),
            desc="Explaining instances",
            total=len(input_tensor) // self.classification_batch_size,
        ):
            batch = input_tensor[i : i + self.classification_batch_size].to(self.model.device)
            if self.attention_type == "ATM":
                attention_weights, _ = self.model.get_attention_weights(batch)
                output[i : i + self.classification_batch_size] = attention_weights
            elif self.attention_type == "ATF":
                attention_weights, _ = self.model.get_attention_flows(batch)
                output[i : i + self.classification_batch_size] = attention_weights

        output = output.cpu()
        if save_to_pickle:
            self.save(output, save_path)
        return output
