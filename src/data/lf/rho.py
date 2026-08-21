import numpy as np
import torch

# NOTE: keep this module free of heavy imports (hydra, wandb, lightning) so that the lightweight
# `cmf` benchmark tool can use the memory-function registry directly.

# # Note: nonsmooth_periodic is already suitable for sequence inputs if t is a PyTorch tensor
# def nonsmooth_periodic_vectorized(t):
#     return (t % 2).float()  # Element-wise modulus operation


class BaseRho:
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    registry = {}

    @classmethod
    def register(cls, name):
        """_summary_

        Args:
            name (_type_): _description_
        """

        def inner_wrapper(wrapped_class):
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @staticmethod
    def create(name, dim=None, T=None, **kwargs):
        """_summary_

        Args:
            name (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if name in BaseRho.registry:
            rho_class = BaseRho.registry[name]
            rho_class.dim = dim
            return rho_class(T=T, **kwargs)
        else:
            raise ValueError(f"No rho registered under name: {name}")


@BaseRho.register("exponential")
class Exponential(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, T=None, alpha=0.1):
        """_summary_

        Args:
            alpha (float, optional): _description_. Defaults to 0.1.
        """
        self.name = "exponential"
        self.param = {"alpha": alpha}

        self.alpha = torch.tensor(alpha).unsqueeze(-1) + 1e-8

        self.T = T
        assert self.dim == self.alpha.shape[0]

    def __call__(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.exp(-t / self.alpha)


@BaseRho.register("polynomial")
class Polynomial(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, T=None, alpha=0.1):
        self.name = "polynomial"
        self.alpha = torch.tensor(alpha).unsqueeze(-1) + 1e-8
        assert self.dim == self.alpha.shape[0]
        self.T = T

    def __call__(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 / (1 + t) ** (1 / self.alpha)


@BaseRho.register("periodic")
class SmoothPeriodic(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, T=None, f=1):
        self.name = "periodic"
        self.f = f

    def __call__(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.sin(2 * np.pi * self.f * t)


@BaseRho.register("impulse")
class impulse(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, T=None, alpha=1):
        self.name = "impulse"

        self.THRESHOLD = 0.5

        self.alpha = torch.floor(torch.tensor(alpha).unsqueeze(-1) * (T // 2))

        assert self.dim == self.alpha.shape[0]

    def __call__(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        return ((t > self.alpha - self.THRESHOLD) & (t < self.alpha + self.THRESHOLD)).float()


@BaseRho.register("airy")
class Airy(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, T=None, alpha=2.25):
        self.name = "airy"
        self.alpha = alpha
        self.T = T

    def __call__(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        from scipy import special

        ai, aip, bi, bip = special.airy(40 * (t / self.T - 0.5 * self.alpha))

        return ai.to(torch.float32)


@BaseRho.register("random")
class Random(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, T=None, seed=1):
        self.name = "random"

        self.seed = seed

    def __call__(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        from src.utils.utils import TempSeed  # lazy: src.utils pulls in hydra/wandb

        with TempSeed(self.seed):
            y = torch.rand(t.shape)

        return y


@BaseRho.register("step")
class Step(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, c=1):
        self.name = "step"
        self.THRESHOLD = 0.5

        self.c = c

    def __call__(self, t):
        return (t < self.c + self.THRESHOLD).float()


@BaseRho.register("test")
class Test(BaseRho):
    """_summary_

    Args:
        BaseRho (_type_): _description_
    """

    def __init__(self, c):
        self.name = "test"
        self.c = c

    def __call__(self, t):
        return torch.pow(t, 0.99) + torch.pow(t, 0.9) + torch.pow(t, 0.7)


@BaseRho.register("mix_exp_impulse")
class MixExpImpulse(BaseRho):
    """(1-beta)*rho_exp(alpha_exp) + beta*rho_impulse(alpha_imp).

    The external 'alpha' parameter acts as beta (mixture weight, swept in experiments).
    alpha_exp and alpha_imp are fixed at the max-alpha values from existing sweep configs.
    beta=0 → pure exponential; beta=1 → pure impulse.
    """

    THRESHOLD = 0.5

    def __init__(self, T=None, alpha=0.5, alpha_exp=0.42857139, alpha_imp=1.0, **kwargs):
        self.beta = torch.tensor(float(alpha)).unsqueeze(-1)
        self.exp_alpha = torch.tensor(float(alpha_exp)).unsqueeze(-1) + 1e-8
        self.imp_pos = torch.floor(torch.tensor(float(alpha_imp)) * (T // 2))

    def __call__(self, t):
        rho_exp = torch.exp(-t / self.exp_alpha)
        rho_imp = (
            (t > self.imp_pos - self.THRESHOLD) & (t < self.imp_pos + self.THRESHOLD)
        ).float()
        return (1 - self.beta) * rho_exp + self.beta * rho_imp


@BaseRho.register("prony")
class Prony(BaseRho):
    """K-mode Prony series with spread parameter gamma.

    lambda_k = exp(-10^(gamma*(k-1)/(K-1)) / T),  k = 1..K
    rho(s)   = (1/K) * sum_k  lambda_k^s

    gamma=0: all modes identical -> single exponential (easy for RNN)
    gamma=1: modes span many timescales -> polynomial-like (hard for RNN)
    Special case K=1: exponent fixed at 0 -> lambda = exp(-1/T).
    """

    def __init__(self, T=None, K=8, gamma=0.5, **kwargs):
        K = int(K)
        gamma = float(gamma)
        k_idx = torch.arange(K, dtype=torch.float32)
        if K == 1:
            exponents = torch.zeros(1)
        else:
            exponents = gamma * k_idx / (K - 1)
        decay = torch.pow(torch.tensor(10.0), exponents) / T
        self.lambdas = torch.exp(-decay)  # (K,)

    def __call__(self, t):
        # t: (D, seq_len)
        log_l = torch.log(self.lambdas)  # (K,)
        powers = torch.exp(log_l.unsqueeze(-1) * t[0].unsqueeze(0))  # (K, seq_len)
        rho_1d = powers.mean(0)  # (seq_len,)
        return rho_1d.unsqueeze(0).expand(t.shape[0], -1)  # (D, seq_len)
