"""Linear-functional (LF) data generation: targets are a memory-function convolution of the inputs.

    y_t = output_activation( sum_{s<=t} rho(s) * inner_activation(x_{t-s}) )

This module only depends on torch (plus the memory functions in ``rho.py``), so it can be used
both by the Lightning ``LFDataModule`` and by the lightweight ``cmf`` benchmark tool.
"""

import logging
import os
import random

import numpy as np
import torch
from torch.fft import irfft, rfft
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from .rho import BaseRho

log = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (same effect as ``lightning.seed_everything``)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


activations = {
    "identity": lambda x: x,
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
    "relu": lambda x: torch.relu(x),
    "softplus": lambda x: torch.nn.functional.softplus(x)
}

def convolve_ref(inputs, rho, dt):
    """_summary_

    Args:
        inputs (_type_): _description_
        rho (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """
    # inputs B * T * D
    T = inputs.shape[1]

    outputs = []
    for t in range(T):
        output = 0
        for s in range(t + 1):
            output += inputs[:, t - s, :] @ rho[:, s].unsqueeze(-1)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=1)

    return outputs, None


def convolve_fft(inputs, rho, dt):
    """_summary_

    Args:
        inputs (_type_): _description_
        rho (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """
    seq_length, input_dim = inputs.shape[1], inputs.shape[2]

    # Assuming rho is a callable function that can be applied in a vectorized manner
    # Create the rho values for the entire sequence length
    time_steps = torch.arange(seq_length, dtype=torch.int64) * dt
    time_steps = time_steps.repeat(input_dim, 1)

    rho_vals = rho(time_steps)

    # Normalize of $\rho$
    # rho_vals = rho_vals / torch.linalg.norm(rho_vals, axis=-1).unsqueeze(-1)

    # rho_vals:  shape (dim, length)

    l_max = inputs.shape[1]
    ud = rfft(F.pad(inputs, pad=(0, 0, 0, l_max, 0, 0)), dim=1)
    Kd = rfft(F.pad(rho_vals, pad=(0, l_max)), dim=-1)
    return (
        torch.sum(
            irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1), axis=-1, keepdim=True
        ),
        rho_vals,
    )


class LF:
    """_summary_"""

    def __init__(
        self,
        data_dir: str,
        N: int,
        T: int,
        D: int,
        rho_name: str,
        rho_param: dict = None,
        dt: float = 1,
        inner_activation: str = "identity",
        output_activation: str = "identity",
        save_file: bool = False,
        input_generator: str = "randn",
        processing: str = None,
        manual_seed=None,
    ):
        if manual_seed is not None:
            _seed_everything(manual_seed)

        self.data_dir = data_dir
        self.N = N
        self.T = T
        self.D = D
        self.rho_name = rho_name
        self.rho_param = rho_param
        self.rho = BaseRho.create(rho_name, dim=self.D, T=self.T, **rho_param)

        self.save_file = save_file

        # This is fixed to be 1.0
        self.dt = dt
        self.input_generator = input_generator

        self.processing = processing

        self.inner_activation = activations[inner_activation]
        self.output_activation = activations[output_activation]

        


    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"T={self.T!r}, "
            f"D={self.D!r}, "
            f"rho_name={self.rho_name!r}, "
            f"rho_param={self.rho_param!r}, "
            f"dt={self.dt!r})"
        )

    def _generate_gaussian(self, num, seq_length, dim):
        """_summary_

        Args:
            num (_type_): _description_
            seq_length (_type_): _description_
            dim (_type_): _description_

        Returns:
            _type_: _description_
        """
        import numpy as np
        from sklearn.metrics.pairwise import rbf_kernel

        xs = np.arange(seq_length) * 0.1
        mean = [0 for _ in xs]
        gram = rbf_kernel(xs[:, np.newaxis], gamma=0.1)

        ys = np.random.multivariate_normal(mean, gram, size=(num, dim))
        return ys.transpose(0, 2, 1)

    def dataset_construction(self, input_data=None) -> TensorDataset:
        """_summary_

        Raises:
            NameError: _description_

        Returns:
            TensorDataset: _description_
        """
        log.info("Start generating data.")
        
        if input_data is not None:
            assert input_data.shape == (self.N, self.T, self.D), "Input data shape mismatch."
            inputs = input_data
        elif self.input_generator == "randn":
            inputs = torch.randn(self.N, self.T, self.D, dtype=torch.float32)
        elif self.input_generator == "rand":
            inputs = torch.rand(self.N, self.T, self.D, dtype=torch.float32)
        elif self.input_generator == "gaussian":
            inputs = torch.tensor(
                self._generate_gaussian(self.N, self.T, self.D), dtype=torch.float32
            )
        else:
            raise NameError(
                f"Input generator type {self.input_generator} not exists, choose from 'rand', 'randn' or 'gaussian'. "
            )

        inputs = self.inner_activation(inputs)


        targets, self.rho_vals = convolve_fft(inputs, self.rho, self.dt)


        targets = self.output_activation(targets)
        

        ####### Add noise
        if self.processing is not None:
            for process in self.processing:
                if process == "noise":
                    ## Add a noise at randome position
                    B, L, D = targets.shape
                    random_indices = torch.randint(0, L, (B,))
                    # Generate random noise for each batch and each feature in D
                    noise = 10  # Random noise for each batch and each feature
                    # Use advanced indexing to add noise at the random indices along L for each batch
                    inputs[torch.arange(B), random_indices, :] += noise

                elif process == "smooth_noise":
                    kernel_size = self.T // 10
                    noise = torch.randn(self.N, self.D, self.T)  # [batch, dim, seq_len]
                    kernel = torch.ones(1, 1, kernel_size) / kernel_size
                    smooth_noise = F.conv1d(noise, kernel, padding=kernel_size // 2, groups=self.D)
                    smooth_noise = smooth_noise.permute(0, 2, 1)  # [batch, seq_len, dim]
                    targets = targets + smooth_noise[:, : self.T, :]
                else:
                    raise NotImplementedError(f"Processing {process} not implemented.")

        dataset = TensorDataset(inputs, targets)
        log.info("Data generation finished.")

        if self.save_file:
            # Construct the datadir
            data_dir_combined = (
                self.data_dir
                + f"/LF_{self.rho.name}_{self.rho.param}__N{self.N}_T{self.T}_D{self.D}_seed{torch.initial_seed()}.pt"
            )
            os.makedirs(self.data_dir, exist_ok=True)
            torch.save(dataset, data_dir_combined)
            log.info(f"Dataset saved to {data_dir_combined}")

        else:
            log.info("Dataset not saved!")

        return dataset, self.rho_vals
