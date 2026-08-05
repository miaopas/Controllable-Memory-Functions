import sys
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """

    # If there is a sweeper option in cfg then clear the default parameters.
    if cfg.get("data") and cfg.data.get("rho_param"):
        cfg = cfg.copy()
        cfg.data.rho_param = cfg.data.rho_param

    if cfg.get("sweeper"):
        for key in cfg.sweeper.param:
            OmegaConf.update(cfg, key, "# See Sweeper!", merge=False)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

        if not cfg.get("skip_confirm"):
            # Check the config and decide to continue
            choice = input("Check the configs, press (ENTER) to continue:")
            if choice == "":
                print("Continuing...")
            else:
                print("Exiting...")
                sys.exit()

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(**kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metric_dict, object_dict = None, None
        # execute the task
        try:
            result = task_func(**kwargs)
            if result is not None:
                metric_dict, object_dict = result

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {kwargs['cfg'].paths.run_dir}")
            import sys

            class SafeStderr:
                def __init__(self, wrapped):
                    self._wrapped = wrapped

                def __getattr__(self, name):
                    return getattr(self._wrapped, name)

                def isatty(self):
                    return False

            sys.stderr = SafeStderr(sys.stderr)
            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb
                
                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def check_and_update_wandb_project(cfg: DictConfig) -> None:
    """Check if W&B project name already exists and append numbers if it does.

    :param cfg: A DictConfig object containing the config tree.
    """
    if not cfg.get("logger") or not cfg.logger.get("wandb"):
        return

    try:
        import wandb

        project_name = cfg.logger.wandb.get("project")
        entity = cfg.logger.wandb.get("entity")

        if not project_name:
            return

        # Get W&B API
        api = wandb.Api()

        # Try to access projects for the entity
        try:
            if entity:
                projects = api.projects(entity=entity)
            else:
                projects = api.projects()

            existing_project_names = [p.name for p in projects]

            # Check if project name exists
            original_name = project_name
            counter = 1

            while project_name in existing_project_names:
                project_name = f"{original_name}_{counter}"
                counter += 1

            # Update config if name was changed
            if project_name != original_name:
                log.info(f"Project '{original_name}' already exists. Using '{project_name}' instead.")
                OmegaConf.update(cfg, "logger.wandb.project", project_name)

        except Exception as e:
            # If we can't access projects (e.g., not logged in), just continue
            log.warning(f"Could not check existing W&B projects: {e}")

    except ImportError:
        # W&B not installed, skip check
        pass


class TempSeed:
    def __init__(self, new_seed):
        self.new_seed = new_seed
        self.old_seed = torch.initial_seed()

    def __enter__(self):
        torch.manual_seed(self.new_seed)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.manual_seed(self.old_seed)
