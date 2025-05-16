from torch import nn
import torch
import math


class PositionalEmbedding(nn.Module):
    """
    Implements positional encoding as described in 'Attention is All You Need'.

    Creates sinusoidal position embeddings that can be added to input embeddings.

    Args:
        d_model (int): Dimension of the model
        max_len (int, optional): Maximum sequence length. Defaults to 5000
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class LatentEmbedding(nn.Module):
    """
    Combines latent embeddings with positional encodings.

    Creates learnable embeddings for input tokens and adds positional information.

    Args:
        input_size (int): Size of input vocabulary
        d_model (int): Dimension of the embeddings
        seq_len (int, optional): Maximum sequence length. Defaults to 512
    """

    def __init__(self, input_size: int, d_model: int, seq_len: int = 512) -> None:
        super().__init__()
        self.positional_embedding = PositionalEmbedding(
            d_model=d_model, max_len=seq_len
        )
        self.latent_embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=d_model
        )
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        x_embed = self.latent_embedding(x) + self.positional_embedding(x)
        return x_embed


class TokenEmbedding(nn.Module):
    """
    Convolutional token embedding layer.
    
    Converts input tokens into embeddings using 1D convolutions with circular padding.
    Initializes weights using Kaiming initialization.

    Args:
        c_in (int): Number of input channels
        d_model (int): Dimension of the output embeddings
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed positional embedding layer with sinusoidal encodings.
    
    Creates non-trainable embeddings using sinusoidal functions at different frequencies,
    similar to the original Transformer paper's positional encodings.

    Args:
        c_in (int): Input size/number of positions
        d_model (int): Dimension of the embeddings
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding layer for time-series data.
    
    Creates embeddings for different time features (minute, hour, weekday, day, month)
    using either fixed or learned embeddings. Combines these embeddings to represent
    temporal information.

    Args:
        d_model (int): Dimension of the embeddings
        embed_type (str): Type of embedding ('fixed' or learned). Defaults to 'fixed'
        freq (str): Time frequency ('h' for hourly, 't' for minutes). Defaults to 'h'
    """
    def __init__(self, d_model: int, embed_type: str = "fixed", freq: str = "h"):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Linear embedding layer for time features.
    
    Projects time features to embedding space using a linear transformation.
    Different input dimensions are used based on the frequency of the time series.

    Args:
        d_model (int): Dimension of the output embeddings
        freq (str): Time frequency ('h', 't', 's', 'm', 'a', 'w', 'd', 'b'). Defaults to 'h'
    """
    def __init__(self, d_model, freq="h"):
        super().__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Combines value embeddings with positional and temporal embeddings.
    Code is based on: Time Series Library (TSLib) https://github.com/thuml/Time-Series-Library

    Creates a combined embedding that includes:
    - Value embeddings from input features
    - Positional encodings for sequence position
    - Optional temporal embeddings for time-series data

    Args:
        c_in (int): Number of input channels/features
        d_model (int): Dimension of the model embeddings
        embed_type (str): Type of embedding ('fixed' or 'learned')
        freq (str): Frequency of temporal features ('h', 't', etc.)
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TimeFeatureEmbedding(d_model=d_model, freq=freq)
            if embed_type == "timeF"
            else TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)
