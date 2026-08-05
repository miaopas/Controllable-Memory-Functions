from typing import Tuple, Optional
import torch
from .tensor_datamodule import TensorDataModule
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    TensorDataset,
    random_split,
)
precision_map = {
    "16-mixed": torch.float16,
    "bf16-mixed": torch.bfloat16,
    "32-true": torch.float32,  # Default
    "64-true": torch.float64
}
class TSDataModule(TensorDataModule):
    """`LightningDataModule` for the LF dataset.

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        teacher,
        train_val_test_split: Tuple[int, int, int] = (16_384, 2_048, 2_048),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        T: int = 128,
        D: int = 1,
        input_generator: str = "rand",
    ) -> None:
        """Initialize a `LFData `.

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

        self.N = sum(train_val_test_split)
        self.T = T
        self.D = D
        self.input_generator = input_generator
        self.teacher = teacher
        self.precision = None

    def prepare_data(self) -> TensorDataset:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        if self.trainer is not None:
            self.precision = self.trainer.precision

        print(f"Precision in DataModule: {self.precision}")
        torch_dtype = precision_map.get(self.precision, torch.float32)
        if self.input_generator == "randn":
            inputs = torch.randn((self.N, self.T, self.D), dtype=torch_dtype)
        elif self.input_generator == "rand":
            inputs = torch.rand((self.N, self.T, self.D), dtype=torch_dtype)
        else:
            raise NameError(
                f"Input generator type {self.input_generator} not exists, choose from 'rand', 'randn' or 'gaussian'. "
            )
        self.teacher.to(torch_dtype)

        torch.save(self.teacher, "data/teacher.pt")
    
        outputs = self.teacher(inputs).detach()
        self.dataset = TensorDataset(inputs, outputs)
        
  