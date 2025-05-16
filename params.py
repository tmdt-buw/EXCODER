from typing import Literal, Union
from data_loader.data_module import WeldingDataModule, CNCDataModule, ECGDataModule

DATASET_NAMES = Literal["Welding", "CNC_Machining", "ECG", "UEA"]
DATASET_MODULES = Union[WeldingDataModule, CNCDataModule, ECGDataModule]
MODEL_NAMES = Literal["DLinear", "TimesNet", "VQ-VAE", "DVAE", "MLP", "VQ-VAE_MLP", "DVAE_MLP", "VQ-VAE_Transformer", "DVAE_Transformer", "TS_Transformer", "SAX_MLP"]
XAI_METHODS = Literal["LIME", "RISE", "SM", "ATM", "SHAP", "IG", "DeepLIFT", "ATF"]
