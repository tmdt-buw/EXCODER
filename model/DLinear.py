import torch
import torch.nn as nn
from model.classification_base import ClassificationLightningModule
from params import DATASET_NAMES


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(ClassificationLightningModule):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    Code is based on: Time Series Library (TSLib) https://github.com/thuml/Time-Series-Library
    """

    def __init__(
        self,
        input_size,
        in_dim,
        num_class,
        individual: bool = False,
        kernel_size: int = 3,
        moving_avg: int = 3,
        learning_rate: float = 1e-3,
        hyperparams_search_str: str = "NoHyperparamSearch",
        dataset_name: DATASET_NAMES = "Welding",
        hash_model: str = "",
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_class,
            in_dim=in_dim,
            d_model=0,
            n_hidden_layers=0,
            dropout_p=0.0,
            learning_rate=learning_rate,
            hyperparams_search_str=hyperparams_search_str,
            dataset_name=dataset_name,
            hash_model=hash_model,
        )
        self.individual = individual
        self.kernel_size = kernel_size
        self.moving_avg = moving_avg

        # Series decomposition block from Autoformer
        self.decompsition = SeriesDecomposition(self.moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for i in range(self.in_dim):
                self.linear_seasonal.append(nn.Linear(input_size, input_size))
                self.linear_trend.append(nn.Linear(input_size, input_size))

                self.linear_seasonal[i].weight = nn.Parameter(
                    (1 / input_size) * torch.ones([input_size, input_size])
                )
                self.linear_trend[i].weight = nn.Parameter(
                    (1 / input_size) * torch.ones([input_size, input_size])
                )
        else:
            self.linear_seasonal = nn.Linear(input_size, input_size)
            self.linear_trend = nn.Linear(input_size, input_size)

            self.linear_seasonal.weight = nn.Parameter(
                (1 / input_size) * torch.ones([input_size, input_size])
            )
            self.linear_trend.weight = nn.Parameter(
                (1 / input_size) * torch.ones([input_size, input_size])
            )
        self.projection = nn.Linear(in_dim * input_size, num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.input_size],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.input_size],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.in_dim):
                seasonal_output[:, i, :] = self.linear_seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def classification(self, x_enc: torch.Tensor):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc: torch.Tensor):
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]
