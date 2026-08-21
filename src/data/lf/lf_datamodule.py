import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    TensorDataset,
    random_split,
)

from src.utils import pylogger

from ..tensor_datamodule import TensorDataModule
from .generator import LF, activations, convolve_fft, convolve_ref  # noqa: F401  (re-exported)
from .rho import BaseRho  # noqa: F401

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class LFDataModule(TensorDataModule):
    """`LightningDataModule` for the LF dataset.

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (16_384, 2_048, 2_048),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        T: int = 128,
        D: int = 1,
        rho_name: str = "exponential",
        rho_param: dict = {"alpha": 0.1},
        inner_activation: str = "identity",
        output_activation: str = "identity",
        dt: float = 1,
        save_file: bool = False,
        input_generator: str = "randn",
        processing=None,
    ) -> None:
        """Initialize a `LFDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.

        :param T: Sequence length. Defaults to `128`.
        :param D: Sequence dimension. Defaults to `1`.
        :param rho_smooth: The smoothness of the pattern. Defaults to `"smooth"`.
        :param rho_pattern: Memory pattern. Defaults to `"short"`.
        """

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        super().__init__(train_val_test_split, batch_size, num_workers, pin_memory)

        self.lf = LF(
            data_dir,
            N=sum(train_val_test_split),
            T=T,
            D=D,
            rho_name=rho_name,
            rho_param=rho_param,
            dt=dt,
            save_file=save_file,
            input_generator=input_generator,
            processing=processing,
            inner_activation= inner_activation,
            output_activation= output_activation,
        )

    def prepare_data(self) -> TensorDataset:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self.dataset, self.rho_vals = self.lf.dataset_construction()


  