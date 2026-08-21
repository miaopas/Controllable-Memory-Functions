"""Built-in reference models (baselines).

Each builtin is a factory ``factory(width, **kwargs) -> nn.Module`` that follows the model contract
in :mod:`cmf.spec`: ``(batch, T, width) -> (batch, T, width)``, causal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn


@dataclass(frozen=True)
class Builtin:
    display_name: str
    factory: Callable[..., nn.Module]
    description: str


def _lstm(width: int, num_layers: int = 1) -> nn.Module:
    return nn.LSTM(width, width, num_layers=num_layers, batch_first=True)


def _gru(width: int, num_layers: int = 1) -> nn.Module:
    return nn.GRU(width, width, num_layers=num_layers, batch_first=True)


def _tcn(width: int, n_layers: int = 8, kernel_size: int = 2, dropout: float = 0.1) -> nn.Module:
    # Reuse the paper's TCN implementation (dilations 1, 2, 4, ... -> receptive field 2**n_layers).
    from src.modules.layers.tcn import TemporalConvNet

    return TemporalConvNet(
        input_size=width, n_layers=n_layers, channel=width, kernel_size=kernel_size, dropout=dropout
    )


class CausalTransformer(nn.Module):
    """Small decoder-only Transformer with sinusoidal positions and a causal mask."""

    def __init__(self, width: int, n_layers: int = 1, n_heads: int = 0, ff_mult: int = 2,
                 max_len: int = 4096):
        super().__init__()
        if n_heads <= 0:  # largest of 4, 2, 1 that divides the width
            n_heads = next(h for h in (4, 2, 1) if width % h == 0)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=ff_mult * width, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        pe = torch.zeros(max_len, width)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: width // 2])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.encoder(x + self.pe[:T], mask=mask, is_causal=True)


def _transformer(width: int, n_layers: int = 1, n_heads: int = 0) -> nn.Module:
    return CausalTransformer(width, n_layers=n_layers, n_heads=n_heads)


BUILTINS: Dict[str, Builtin] = {
    "lstm": Builtin("LSTM", _lstm, "torch.nn.LSTM, 1 layer"),
    "gru": Builtin("GRU", _gru, "torch.nn.GRU, 1 layer"),
    "tcn": Builtin("TCN", _tcn, "dilated causal CNN from the paper (src/modules/layers/tcn.py), 8 levels"),
    "transformer": Builtin("Transformer", _transformer, "1-layer causal Transformer, sinusoidal positions"),
}
