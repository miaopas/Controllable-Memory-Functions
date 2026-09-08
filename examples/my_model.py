"""Example: benchmark your own model.

A model only needs to map (batch, T, m) -> (batch, T, m) causally. Here: a GRU with a gated
residual connection, but anything works (SSMs, convolutions, attention, ...).

Command line (from the repository root, after ``pip install -e .``):

    cmf check  --model examples.my_model:ResidualGRU --size-arg width
    cmf run    --model examples.my_model:ResidualGRU --size-arg width --suite quick --workers 8
    cmf report runs/ResidualGRU_quick --baselines all --out reports/residual_gru.html

Or run this file, which does the same from Python and compares against the bundled baselines:

    python examples/my_model.py --suite smoke
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ResidualGRU(nn.Module):
    def __init__(self, width: int, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(width, width, num_layers=num_layers, batch_first=True)
        self.gate = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return x + torch.sigmoid(self.gate(x)) * h


if __name__ == "__main__":
    import cmf

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    mine = cmf.run(ResidualGRU, name="ResidualGRU", suite=args.suite, sizes=args.sizes,
                   workers=args.workers)
    out = f"reports/ResidualGRU_{args.suite}"
    path = cmf.report([mine], out=f"{out}.html", pdf=f"{out}.pdf", baselines="all")
    print(f"Open {path} in a browser (PDF: {out}.pdf).")
