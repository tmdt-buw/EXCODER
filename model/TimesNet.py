import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from model.embedding import DataEmbedding
from model.classification_base import ClassificationLightningModule
from params import DATASET_NAMES


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
        top_k: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(ClassificationLightningModule):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    Code is based on: Time Series Library (TSLib) https://github.com/thuml/Time-Series-Library
    """

    def __init__(
        self,
        input_size: int,
        in_dim: int,
        num_class: int,
        d_model: int,
        n_hidden_layers: int,
        dropout_p: float,
        learning_rate: float,
        freq: str = "h",
        embed_type: str = "timeF",
        d_ff: int = 4,
        num_kernels: int = 6,
        top_k: int = 2,
        hyperparams_search_str: str = "NoHyperparamSearch",
        dataset_name: DATASET_NAMES = "Welding",
        hash_model: str = "",
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_class,
            in_dim=in_dim,
            d_model=d_model,
            n_hidden_layers=n_hidden_layers,
            dropout_p=dropout_p,
            learning_rate=learning_rate,
            hyperparams_search_str=hyperparams_search_str,
            dataset_name=dataset_name,
            hash_model=hash_model
        )

        self.label_len = num_class
        self.pred_len = num_class
        
        self.enc_embedding = DataEmbedding(
            c_in=in_dim,
            d_model=d_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout_p,
        )

        self.model = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=input_size,
                    pred_len=0,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_kernels=num_kernels,
                    top_k=top_k,
                )
                for _ in range(n_hidden_layers)
            ]
        )
        self.layer = n_hidden_layers
        self.layer_norm = nn.LayerNorm(d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout_p)
        self.projection = nn.Linear(
            d_model * input_size, num_class
        )

    def classification(self, x_enc: torch.Tensor):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
            enc_out = self.layer_norm(enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)

        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x: torch.Tensor):
        dec_out = self.classification(x)
        return dec_out  # [B, N]