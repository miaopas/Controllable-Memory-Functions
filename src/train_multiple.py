import os
import random
import shutil
import string
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hydra
import ray
import rootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# This disables printing logs from individual trials.
random_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

ray.init(log_to_driver=False, _temp_dir=f"/tmp/ray_{random_string}")

from src.utils import (
    RankedLogger,
    check_and_update_wandb_project,
    extras,
    get_metric_value,
    initialize_hydra_resolvers,
    initialize_resolvers,
    process_results_after_training,
    setup_training_components,
    suppress_logging,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
initialize_resolvers()


@task_wrapper
def train_wrapper(config: Dict = None, cfg: DictConfig = None, hydra_config: HydraConfig = None):
    """_summary_

    Args:
        config (Dict, optional): _description_. Defaults to None.
        cfg (DictConfig, optional): _description_. Defaults to None.
        hydra_config (HydraConfig, optional): _description_. Defaults to None.
    """
    # Initialize HydraConfig for subprocesses
    initialize_hydra_resolvers(hydra_config)
    initialize_resolvers()

    name = Path(session.get_trial_dir()).name

    # Update sweeped cfg params
    cfg = cfg.copy()
    for key, value in config.items():
        if "rho_param" in key:
            cfg.data.rho_param = cfg.data.rho_param
        OmegaConf.update(cfg, key, value)

    output_dir = Path(cfg.paths.output_dir)
    ray_dir = Path(ray.train.get_context().get_trial_dir()).parts[-2:]
    ray_dir = Path(*ray_dir)
    OmegaConf.update(cfg, "paths.run_dir", str(output_dir / ray_dir) + "_out")

    # Setup all training components using shared utility
    object_dict = setup_training_components(cfg, pass_cfg_to_model=True)

    # Extract components for use in training
    datamodule = object_dict["datamodule"]
    model = object_dict["model"]
    callbacks = object_dict["callbacks"]
    logger = object_dict["logger"]
    trainer = object_dict["trainer"]

    # Add extra call_backs
    tune_report_callback = TuneReportCheckpointCallback(
        metrics={
            "train/loss": "train/loss_epoch",
            "val/loss": "val/loss",
            "lr": "lr",
            "rLoss": "train/rLoss_epoch",
        },
        save_checkpoints=False,
        on="validation_end",
    )
    trainer.callbacks.append(tune_report_callback)

    # Add trial_name to the object_dict that was created by setup_training_components
    object_dict["trial_name"] = name

    trainer.fit(model=model, datamodule=datamodule)


def train(
    config: Dict, cfg: DictConfig, hydra_config: HydraConfig
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    train_wrapper(config=config, cfg=cfg, hydra_config=hydra_config)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_multiple.yaml")
def tune_model(cfg: DictConfig) -> Optional[float]:
    """_summary_
    name, model, config, parameter_columns, epochs, resources_per_trial
    Args:
        name (str): Name of this run
        model (Model):The model
        data (tuple): (input_loader, output_loader)
        train_test_split (float): ratio of train test split
    """

    if not cfg.get("sweeper"):
        raise ValueError("Sweeper not specified!")

    if cfg.get("no_merging_wandb_projects"):
        check_and_update_wandb_project(cfg)

    extras(cfg)

    reporter = CLIReporter(
        parameter_columns=["model/hid_dim", "lr"],
        metric_columns=["train/loss", "val/loss"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train, cfg=cfg, hydra_config=HydraConfig.get_state()
    )
    trainable_with_gpu = tune.with_resources(
        train_fn_with_parameters, resources=dict(cfg.sweeper.resources)
    )

    # Initialize param spaces
    param_conf = {}

    for key, value in cfg.sweeper.param.items():
        if isinstance(value, Iterable) and not isinstance(value, str):
            param_conf[key] = tune.grid_search(value)
        else:
            try:
                # Need to import modules if param defined by other iteratbles
                import numpy as np

                param_conf[key] = tune.grid_search(eval(value))
            except Exception as e:
                print(e)

    directory_path = os.path.expanduser("~/ray_results/" + cfg.run_specs.task_name + "_ray")
    try:
        shutil.rmtree(directory_path)
        log.info(f"The directory {directory_path} has been removed successfully.")
    except Exception as e:
        log.error(f"Error: {e}")

    tuner = tune.Tuner(
        trainable_with_gpu,
        run_config=air.RunConfig(
            name=cfg.run_specs.task_name + "_ray",
            storage_path=cfg.paths.output_dir,
            progress_reporter=reporter,
            verbose=1,
            log_to_file=True,
        ),
        param_space=param_conf,
    )

    log.info("Starting Ray Tune training...")
    tuner.fit()
    log.info("Ray Tune training finished!")

    # Process results after training completes
    log.info("Starting result processing...")
    process_results_after_training(cfg)
    log.info("Result processing completed!")


if __name__ == "__main__":
    tune_model()
