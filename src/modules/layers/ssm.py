import numpy as np
import torch
from torch import nn
from torch.fft import fft, ifft, irfft, rfft
from torch.nn import functional as F

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def log_vandermonde_naive(v, x, L):
    r"""
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))  # (... N L)
    vandermonde_prod = torch.einsum("... n, ... n l -> ... l", v, vandermonde_matrix)  # (... L)

    return vandermonde_prod


try:
    from src.functionals.vandermonde import log_vandermonde, log_vandermonde_lazy

    has_pykeops = True
    _log_vandermonde = log_vandermonde_lazy
    log.info("<Using Pykeops for fast Vandermonde kernel computation>")

except ImportError:
    has_pykeops = False

    _log_vandermonde = log_vandermonde_naive

    log.warning(
        "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency."
    )


class scanSSM(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mode="conv", A_gain=0.9, layers=1, dtype=torch.float64, A_data=None, A_train=True):
        super().__init__()
        input_dims =[hid_dim]*layers 
        input_dims[0] = input_dim

        if A_data is None:
            A_data = [None]*layers


        self.layers = nn.ModuleList([scanSSMcell(input_dims[i], hid_dim, hid_dim, mode, A_gain, dtype, A_data[i], A_train) for i in range(layers)])

        self.output = nn.Linear(hid_dim, output_dim, bias=False)

        

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x
    


class scanSSMcell(nn.Module):
    """H(t+1) = A h(t) + B x(t) y(t) = C h(t)"""

    # Assume C is eye matrix

    def __init__(self, input_dim, output_dim, hid_dim, mode="conv", A_gain=0.9, dtype=torch.float64, A_data=None, A_train=True):
        super().__init__()

        self.hid_dim = hid_dim
        self.A = nn.Parameter(torch.empty(hid_dim, hid_dim))
        self.B = nn.Parameter(torch.empty(hid_dim, input_dim))
        self.C = nn.Parameter(torch.eye(hid_dim))

        self.mode = mode

        # torch.nn.init.orthogonal_(self.A,gain=A_gain)
        if A_data:
            self.A.data = torch.diag(torch.tensor(A_data))
        else:
            torch.nn.init.orthogonal_(self.A,gain=A_gain)
        
        if not A_train:
            self.A.requires_grad=False

        self.C.requires_grad=False

        torch.nn.init.xavier_uniform_(self.B)
       
        
        self.register_buffer("h0", torch.zeros(hid_dim, 1))

        self=self.to(dtype)


    
    @staticmethod
    def _scan_SSM(Ab, Bb, Cb, x, h0):
        def step(h_k_1, x_k):
            h_k = Ab @ h_k_1 + Bb @ x_k.unsqueeze(-1)
            y_k = Cb @ h_k
            return h_k, y_k.squeeze(-1)

        h_k = h0
        y = []
        for k in range(x.shape[1]):
            x_k = x[:, k, :]
            h_k, y_k = step(h_k, x_k)
            y.append(y_k)
        return torch.stack(y, dim=1)

    @staticmethod
    def _K_conv(Ab, Bb, Cb, L):
        return torch.stack([(Cb @ torch.matrix_power(Ab, l) @ Bb) for l in range(L)],dim=1)

    @staticmethod
    def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        l_max = u.shape[1]
        ud = rfft(F.pad(u.float(), pad=(0, 0, 0, l_max, 0, 0)), dim=1).unsqueeze(1)
        Kd = rfft(F.pad(K.transpose(-2, -1).float(), pad=(0, l_max)), dim=-1).unsqueeze(0)
        y = irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1).type_as(u)

        return torch.sum(y,dim=-1).transpose(-2,-1)

    def _conv_from_gen(self, gen, L):
        # Evaluate at roots of unity
        # Generating function is (-)z-transform, so we evaluate at (-)root
        Omega_L = torch.tensor(np.exp((-2j * np.pi) * (np.arange(L) / L)))

        atRoots = torch.vmap(gen)(Omega_L)
        # Inverse FFT
        out = ifft(atRoots.transpose(0,1), L,dim=1)
        return out.real

    def _K_gen_simple(self, Ab, Bb, Cb, L):
        K = self._K_conv(Ab, Bb, Cb, L)

        def gen(z):
            return torch.sum(K * (z ** torch.arange(L))[None,:,None],dim=1).to(K.device)

        return gen

    def _K_gen_inverse(self, Ab, Bb, Cb, L):
        I = torch.eye(*Ab.size(), out=torch.empty_like(Ab))
        Ab_L = torch.matrix_power(Ab, L)
        Ct = Cb @ (I - Ab_L)

        def gen(z):
            # Determine complex dtype based on Ab's dtype
            complex_dtype = torch.complex64 if Ab.dtype == torch.float32 else torch.complex128

            return (
                Ct.to(complex_dtype).conj()
                @ torch.inverse(I - Ab.to(complex_dtype) * z.to(Ab.device))
                @ Bb.to(complex_dtype)
            )
        return gen

    def forward(self, x, mode=None):
        if mode is not None:
            self.mode = mode
        if self.mode == "scan":
            y = self._scan_SSM(self.A, self.B, self.C, x, self.h0)
        elif self.mode == "conv":
            L = x.shape[1]

            gen = self._K_gen_inverse(self.A, self.B, self.C, L)

            # K: [output_dim,seq_lengeh,input_dim]

            K = self._conv_from_gen(gen, L)

            y = self._non_circular_convolution(x, K)
        else:
            raise Exception("mode not defined")
        return y

class scanSSMcell1(nn.Module):
    """H(t+1) = A h(t) + B x(t) y(t) = C h(t)"""

    def __init__(self, input_dim, output_dim, hid_dim, mode="conv", A_gain=0.9):
        super().__init__()

        self.A = nn.Parameter(torch.empty(hid_dim, hid_dim))
        self.B = nn.Parameter(torch.empty(hid_dim, input_dim))
        self.C = nn.Parameter(torch.empty(output_dim, hid_dim))
        self.mode = mode

        torch.nn.init.orthogonal_(self.A,gain=A_gain)
        torch.nn.init.xavier_uniform_(self.B)
        torch.nn.init.xavier_uniform_(self.C)

        # self.A.requires_grad=False

        # register h0
        self.register_buffer("h0", torch.zeros(hid_dim, 1))

    @staticmethod
    def _scan_SSM(Ab, Bb, Cb, x, h0):
        def step(h_k_1, x_k):
            h_k = Ab @ h_k_1 + Bb @ x_k.unsqueeze(-1)
            y_k = Cb @ h_k
            return h_k, y_k.squeeze(-1)

        h_k = h0
        y = []
        for k in range(x.shape[1]):
            x_k = x[:, k, :]
            h_k, y_k = step(h_k, x_k)
            y.append(y_k)
        return torch.stack(y, dim=1)

    @staticmethod
    def _K_conv(Ab, Bb, Cb, L):
        return torch.stack([(Cb @ torch.matrix_power(Ab, l) @ Bb) for l in range(L)],dim=1)

    @staticmethod
    def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        l_max = u.shape[1]
        ud = rfft(F.pad(u.float(), pad=(0, 0, 0, l_max, 0, 0)), dim=1).unsqueeze(1)
        Kd = rfft(F.pad(K.transpose(-2, -1).float(), pad=(0, l_max)), dim=-1).unsqueeze(0)
        y = irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1).type_as(u)

        return torch.sum(y,dim=-1).transpose(-2,-1)

    def _conv_from_gen(self, gen, L):
        # Evaluate at roots of unity
        # Generating function is (-)z-transform, so we evaluate at (-)root
        Omega_L = torch.tensor(np.exp((-2j * np.pi) * (np.arange(L) / L)))

        atRoots = torch.vmap(gen)(Omega_L)
        # Inverse FFT
        out = ifft(atRoots.transpose(0,1), L,dim=1)
        return out.real

    def _K_gen_simple(self, Ab, Bb, Cb, L):
        K = self._K_conv(Ab, Bb, Cb, L)

        def gen(z):
            return torch.sum(K * (z ** torch.arange(L))[None,:,None],dim=1).to(K.device)

        return gen

    def _K_gen_inverse(self, Ab, Bb, Cb, L):
        I = torch.eye(*Ab.size(), out=torch.empty_like(Ab))
        Ab_L = torch.matrix_power(Ab, L)
        Ct = Cb @ (I - Ab_L)

        def gen(z):
            # print(torch.linalg.svd(I - Ab)[1])
            return (
                Ct.type(torch.complex64).conj()
                @ torch.inverse(I - Ab * z.to(Ab.device))
                @ Bb.type(torch.complex64)
            )

        return gen

    def forward(self, x, mode=None):
        if mode is not None:
            self.mode = mode
        if self.mode == "scan":
            y = self._scan_SSM(self.A, self.B, self.C, x, self.h0)
        elif self.mode == "conv":
            L = x.shape[1]

            gen = self._K_gen_inverse(self.A, self.B, self.C, L)

            # K: [output_dim,seq_lengeh,input_dim]

            K = self._conv_from_gen(gen, L)

            y = self._non_circular_convolution(x, K)
        else:
            raise Exception("mode not defined")
        return y

def initialize_dt(d):
    x = torch.arange(d) / d
    y = (torch.tanh(x * 8 - 2) + 1) / 2

    # y = torch.ones(d)*0.01
    return nn.Parameter(y).unsqueeze(-1)


class diagSSM(nn.Module):
    """H(t+1) = A h(t) + B x(t) y(t) = C h(t)"""

    def __init__(self, input_dim, output_dim, d_model, d_state, mode="scan"):
        super().__init__()

        self.dtype_r = torch.float32
        self.dtype_c = torch.complex64

        self._A = nn.Parameter(torch.empty(d_model, d_state, dtype=self.dtype_r))
        self.B = nn.Parameter(torch.empty(d_model, d_state, input_dim, dtype=self.dtype_c))
        self.C = nn.Parameter(torch.empty(d_model, output_dim, d_state, dtype=self.dtype_c))

        self._dt = nn.Parameter(initialize_dt(d_model).type(self.dtype_r))

        self.mode = mode

        torch.nn.init.uniform_(self._A, a=-2 * np.pi, b=2 * np.pi)
        torch.nn.init.xavier_uniform_(self.B)
        torch.nn.init.xavier_uniform_(self.C)

        self.register_buffer("h0", torch.zeros(d_state, 1, dtype=self.dtype_c))

    @staticmethod
    def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        l_max = u.shape[1]
        ud = rfft(F.pad(u, pad=(0, 0, 0, l_max, 0, 0)), dim=1)
        Kd = rfft(F.pad(K, pad=(0, l_max)), dim=-1)
        return irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1)

    @property
    def A(self) -> torch.Tensor:
        A = (self._A * 1j).type(self.dtype_c)
        return A

    @property
    def v(self) -> torch.Tensor:
        v = self.B.permute(2, 0, 1) * self.C.permute(1, 0, 2)
        v = v * (torch.exp(self.dtA) - 1.0) / self.A
        return v

    @property
    def dtA(self) -> torch.Tensor:
        return self.dt * self.A

    @property
    def dt(self) -> torch.Tensor:
        return torch.clamp(self._dt, min=0.01, max=0.99)

    def forward(self, x):
        L, d = x.shape[1], x.shape[2]

        # assert d == self.B.shape[2]

        v = self.v.contiguous()
        dtA = self.dtA.contiguous()

        K = torch.sum(_log_vandermonde(v, dtA, L), axis=1).real
        y = self._non_circular_convolution(x, K)

        y = torch.sum(y, axis=-1, keepdim=True)

        return y

    def _setup_step(self):
        dtA = self.dtA.contiguous()

        self.dA = torch.exp(dtA)  # (H N)

        scale = ((torch.exp(dtA) - 1.0) / self.A).unsqueeze(-1)
        self.dB = self.B * scale  # (C H N)

        self.dC = self.C

    def step(self, u, state):
        next_state = torch.einsum("h n, b h n -> b h n", self.dA, state) + torch.einsum(
            "h n, b h -> b h n", self.dB, u
        )
        y = torch.einsum("c h n, b h n -> b c h", self.dC, next_state)
        return y.real, next_state
