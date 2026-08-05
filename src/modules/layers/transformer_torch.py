import math

import torch
import torch.nn as nn


class ConstantPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class ParameterPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = nn.Parameter(torch.randn(max_len, d_model))
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
class TransformerModel(nn.Module):
    def __init__(self,
            input_dim,
            output_dim,
            num_layers,
            hid_dim,
            nhead,
            src_length,
            dim_feedforward,
            dropout=0.1,):
        
        super().__init__()
        # hid_dim = hid_dim * nhead
        # hid_dim = hid_dim if hid_dim % 2 == 0 else hid_dim + 1

        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hid_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hid_dim, output_dim)
        
        # Causal Mask
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(src_length, src_length), diagonal=1).bool()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        
        x = self.transformer(x, mask=self.causal_mask)  # (seq_len, batch, d_model)
        
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        return self.output_layer(x)  # (batch, seq_len, output_dim)
