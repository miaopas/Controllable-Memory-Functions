from mamba_ssm import Mamba as MambaBlock
from torch import nn


import torch.nn as nn

class Mamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_state=16,d_conv=4,expand=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        self.network = MambaBlock(d_model=d_model, # Model dimension d_model
                             d_state=d_state,  # SSM state expansion factor
                             d_conv=d_conv,    # Local convolution width
                             expand=expand)

    def forward(self, x):
        """
        Forward pass through Mamba model

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        x = self.input_proj(x)
        x = self.network(x)
        x = self.output_proj(x)
        return x

