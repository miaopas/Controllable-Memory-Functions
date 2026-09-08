"""Compact, dependency-free reference models: S4D and Mamba in pure PyTorch.

These are small re-implementations for benchmarking on CPU or GPU without custom kernels. They follow
the published designs, but are not the official implementations (``state-spaces/s4``,
``state-spaces/mamba``); the paper's own versions live in ``src/modules/layers/``.

Both blocks map ``(batch, T, width) -> (batch, T, width)`` causally, as the cmf model contract requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- S4D
class S4DKernel(nn.Module):
    """Diagonal state-space kernel (S4D-Lin initialisation, ZOH discretisation).

    Gu et al., "On the Parameterization and Initialization of Diagonal State Space Models", 2022.
    """

    def __init__(self, width: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        n = d_state // 2  # complex conjugate pairs
        log_dt = torch.rand(width) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(width, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n).float().repeat(width, 1))
        C = torch.randn(width, n, dtype=torch.cfloat) * (0.5 ** 0.5)
        self.C = nn.Parameter(torch.view_as_real(C))

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)[:, None]                         # (H, 1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag            # (H, N)
        dtA = A * dt
        C = torch.view_as_complex(self.C) * (torch.exp(dtA) - 1.0) / A
        powers = dtA[..., None] * torch.arange(L, device=A.device)   # (H, N, L)
        return 2 * torch.einsum("hn,hnl->hl", C, torch.exp(powers)).real  # (H, L)


class S4D(nn.Module):
    """One pre-norm residual S4D block: x + GLU(Linear(GELU(S4D(LayerNorm(x)))))."""

    def __init__(self, width: int, d_state: int = 64, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([_S4DLayer(width, d_state, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _S4DLayer(nn.Module):
    def __init__(self, width: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.kernel = S4DKernel(width, d_state)
        self.D = nn.Parameter(torch.randn(width))
        self.out = nn.Linear(width, 2 * width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        u = self.norm(x).transpose(1, 2)                              # (B, H, T)
        k = self.kernel(T)                                            # (H, T)
        n = 2 * T                                                     # zero-pad: causal, not circular
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :T]
        y = y + u * self.D[:, None]
        y = self.drop(F.gelu(y)).transpose(1, 2)                      # (B, T, H)
        return x + F.glu(self.out(y), dim=-1)


# --------------------------------------------------------------------------- Mamba
def selective_scan(a: torch.Tensor, b: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """All states of the recurrence h_t = a_t * h_{t-1} + b_t (h_{-1} = 0) along dim 1.

    Exact chunked scan: a log-depth (Hillis-Steele) scan inside chunks of ``chunk`` steps, all
    chunks at once, then a short sequential pass that carries the state across chunks. About 4x
    faster than a plain loop on CPU, without custom kernels.
    """
    B, T = a.shape[:2]
    rest = a.shape[2:]
    pad = (-T) % chunk
    if pad:
        zeros = (0, 0) * len(rest)
        a = F.pad(a, zeros + (0, pad), value=1.0)
        b = F.pad(b, zeros + (0, pad))
    n = a.shape[1] // chunk
    a = a.reshape(B, n, chunk, *rest)
    b = b.reshape(B, n, chunk, *rest)
    d = 1
    while d < chunk:  # combine (a, b) pairs d steps apart; b first, it needs the old a
        b = torch.cat([b[:, :, :d], b[:, :, d:] + a[:, :, d:] * b[:, :, :-d]], dim=2)
        a = torch.cat([a[:, :, :d], a[:, :, d:] * a[:, :, :-d]], dim=2)
        d *= 2
    carry = torch.zeros_like(b[:, 0, -1])
    out = []
    for i in range(n):  # a[:, i] is now the decay since the start of chunk i
        h = b[:, i] + a[:, i] * carry[:, None]
        out.append(h)
        carry = h[:, -1]
    return torch.stack(out, 1).reshape(B, n * chunk, *rest)[:, :T]


class Mamba(nn.Module):
    """Pre-norm residual Mamba (selective SSM) blocks.

    Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023. The scan is
    :func:`selective_scan` (exact, pure PyTorch), slower than the fused CUDA kernel of ``mamba_ssm``.
    """

    def __init__(self, width: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([_MambaLayer(width, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return x


class _MambaLayer(nn.Module):
    def __init__(self, width: int, d_state: int, d_conv: int, expand: int,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        e = expand * width
        self.dt_rank = max(1, math.ceil(width / 16))
        self.d_state = d_state
        self.in_proj = nn.Linear(width, 2 * e)
        self.conv = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv - 1)
        self.x_proj = nn.Linear(e, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, e)
        dt = torch.exp(torch.rand(e) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():  # bias so that softplus(bias) = dt
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)                       # (B, T, E) each
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))  # causal depthwise conv
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                             # (B, T, E)
        A = -torch.exp(self.A_log)                                    # (E, N)
        dA = torch.exp(dt[..., None] * A)                             # (B, T, E, N)
        dBu = (dt * u)[..., None] * Bm[:, :, None, :]                 # (B, T, E, N)
        hs = selective_scan(dA, dBu)                                  # (B, T, E, N)
        y = torch.einsum("bten,btn->bte", hs, Cm) + u * self.D
        return self.out_proj(y * F.silu(z))
