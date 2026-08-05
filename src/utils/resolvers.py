from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import setup_globals
from omegaconf import MISSING, OmegaConf

from ..data.lf.rho import BaseRho


def initialize_hydra_resolvers(hydra_config):
    """This is to initialize hydra default resolvers in subprocesses."""
    HydraConfig.set_state(hydra_config)
    setup_globals()


def initialize_resolvers():
    """This is to initialize customized resolvers."""

    # This resolver checks whether a key exists, if not exists or False, return True
    def negate(a, *, _root_):
        return not _root_.get(a, False)

    OmegaConf.register_new_resolver(
        "not",
        negate,
        replace=True,
    )

    # This resolver checks whether two inputs are equal
    OmegaConf.register_new_resolver(
        "check_equal",
        lambda x, y: True if x == y else False,
        use_cache=True,
        replace=True,
    )

    # This resolvers gets the possible parameters from a given rho name.
    def get_rho_param(name):
        import inspect

        # Now should only have one parameter alpha
        def get_rho_param(cls):
            init_params = inspect.signature(cls.__init__).parameters
            return {param.name: MISSING for param in init_params.values() if param.name != "self"}

        if name in BaseRho.registry:
            rho_class = BaseRho.registry[name]
            rho_param_name = get_rho_param(rho_class)

            return rho_param_name
        else:
            raise ValueError(f"No rho registered under name: {name}")

    OmegaConf.register_new_resolver(
        "get_rho_param",
        get_rho_param,
        use_cache=True,
        replace=True,
    )

    def random_string(length=4):
        import random
        import string
        import time

        random.seed(time.time())
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    OmegaConf.register_new_resolver(
        "random_id",
        random_string,
        use_cache=True,
        replace=True,
    )

    def _import(name, *, _root_):
        return _root_._import_path.get(f"{name}")

    OmegaConf.register_new_resolver(
        "import",
        _import,
        replace=True,
    )

    # Append one item to the list
    def _append_tag(item, *, _root_):
        return _root_.run_specs.tags + item

    OmegaConf.register_new_resolver("append_tag", _append_tag, replace=True)

    # transfer a name with _ into / so that it becomes a path
    OmegaConf.register_new_resolver("to_path", lambda x: x.replace("_", "/"), replace=True)

    def _round(number):
        if isinstance(number, float):
            return round(number, 2)
        else:
            return number

    OmegaConf.register_new_resolver(
        "round",
        _round,
        use_cache=True,
        replace=True,
    )
