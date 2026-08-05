import rootutils
from lightning.pytorch import Callback
import pytorch_lightning as pl
import os
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np
from src.utils import RankedLogger
import torch
import wandb
log = RankedLogger(__name__, rank_zero_only=True)


class CheckUnusedParametersCallback(Callback):
    """_summary_

    Args:
        Callback (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.has_checked_unused_parameters = False

    def on_after_backward(self, trainer, pl_module):
        if self.has_checked_unused_parameters:
            return
        log.info("Training is starting. Checking for unused parameters...")
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is None:
                log.warning(f"Warning: Parameter '{name}' was not used during the forward pass.")
        self.has_checked_unused_parameters = True


class SpectrumLogger(Callback):
    def __init__(self, layer_name: str = "model.layer.weight", log_singular_values=True):
        self.layer_name = layer_name
        self.log_singular_values = log_singular_values

    def on_train_epoch_end(self, trainer, pl_module):
        # Grab the weight matrix
        try:
            weight = dict(pl_module.named_parameters())[self.layer_name].detach().cpu()
        except KeyError:
            print(f"[SpectrumLogger] Could not find layer: {self.layer_name}")
            return

        # Compute spectrum
        if self.log_singular_values:
            spectrum = torch.linalg.svdvals(weight).numpy()
        else:
            spectrum = torch.linalg.eigvals(weight).real.numpy()  # only real parts

        # Log to wandb
        if trainer.logger and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log({
                f"spectrum/{self.layer_name}": wandb.Histogram(spectrum),
                "global_step": trainer.global_step
            })



class S4DSpectrumLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        s4d_module = None
        for module in pl_module.modules():
            if module.__class__.__name__ == "SSMKernelDiag":
                s4d_module = module
                break

        if s4d_module is None:
            print("No S4D module found in model.")
            return

        A = getattr(s4d_module, 'A', None)

        if A is None:
            print("No attribute 'A' in S4D module.")
            return

        A_np = A.detach().cpu().numpy()

        # Save the real and imaginary parts separately
        save_path = f"A_complex_epoch_{trainer.current_epoch}.npz"
        np.savez(save_path, real=np.real(A_np), imag=np.imag(A_np))

        # Upload the file to wandb
        artifact = wandb.Artifact(f"A_complex_epoch_{trainer.current_epoch}", type="s4d_matrix")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

        # Optionally remove the file after logging
        os.remove(save_path)