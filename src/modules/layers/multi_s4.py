"""Multi-layer S4 wrapper for stacking multiple S4Block layers."""

import torch.nn as nn
from src.modules.layers.s4 import S4Block


class MultiS4(nn.Module):
    """Multi-layer S4 wrapper that stacks multiple S4Block layers.

    This wrapper allows easy configuration of multi-layer S4 models while
    maintaining backward compatibility with single-layer S4 configurations.
    When n_layers=1, this behaves identically to a single S4Block.

    Args:
        d_model: Model dimension (must be consistent across all layers)
        n_layers: Number of S4 blocks to stack (default: 1)
        transposed: Whether to use transposed format (default: False)
        **layer_args: Additional arguments passed to each S4Block
    """

    def __init__(
        self,
        d_model,
        n_layers=1,
        transposed=False,
        **layer_args,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.transposed = transposed

        # Create n_layers S4 blocks
        self.layers = nn.ModuleList([
            S4Block(
                d_model=d_model,
                transposed=transposed,
                **layer_args
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, **kwargs):
        """Forward pass through all S4 layers.

        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to S4Block forward

        Returns:
            Output tensor (same format as S4Block output)
        """
        for layer in self.layers:
            x = layer(x, **kwargs)
            # S4Block returns (output, state), we only need output for next layer
            if isinstance(x, tuple):
                x = x[0]

        return x

    @property
    def d_output(self):
        """Output dimension (same as input dimension)."""
        return self.d_model
