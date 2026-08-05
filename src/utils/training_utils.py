from typing import Any, Dict, List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

from src.utils import instantiate_callbacks, instantiate_loggers, log_hyperparameters, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def setup_training_components(cfg: DictConfig, pass_cfg_to_model: bool = False) -> Dict[str, Any]:
    """Setup all training components (datamodule, model, callbacks, logger, trainer).

    This function extracts the common training component instantiation logic
    shared between train.py and train_multiple.py.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
        pass_cfg_to_model: If True, pass cfg as parameter to model instantiation.

    Returns:
        Dict containing all instantiated training components:
        - cfg: The configuration
        - datamodule: LightningDataModule instance
        - model: LightningModule instance
        - callbacks: List of callbacks
        - logger: List of loggers
        - trainer: Trainer instance
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.run_specs.seed:
        # Apply seed_offset if specified (useful for running multiple experiment sets)
        seed_offset = cfg.run_specs.get("seed_offset")
        if seed_offset is not None:
            base_seed = cfg.run_specs.seed
            final_seed = base_seed + seed_offset
            # Update cfg.run_specs.seed to the final value so it gets logged to WandB
            from omegaconf import OmegaConf
            OmegaConf.update(cfg, "run_specs.seed", final_seed)
            log.info(f"[SEED] Applied offset: {base_seed} + {seed_offset} = {final_seed}")

        # Use the (possibly updated) seed value
        L.seed_everything(cfg.run_specs.seed, workers=True)
        log.info(f"[SEED] Set global random seed to: {cfg.run_specs.seed}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.module.lit_module._target_}>")
    if pass_cfg_to_model:
        model: LightningModule = hydra.utils.instantiate(cfg.module.lit_module, cfg=cfg)
    else:
        model: LightningModule = hydra.utils.instantiate(cfg.module.lit_module)

    model.hparams["cfg"] = cfg

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Setup WandB metrics if WandB logger is present
    for lg in logger:
        if isinstance(lg, WandbLogger):
            import wandb

            lg.watch(model, log_freq=100)
            wandb.define_metric("train/rLoss_epoch", summary="min")
            wandb.define_metric("train/rLoss1_epoch", summary="min")
            wandb.define_metric("train/loss_epoch", summary="min")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    return object_dict