from typing import Any, Dict, Sequence, Tuple, Union, List

from dial_mpc.envs.unitree_aliengo_env import (
    UnitreeAliengoEnvConfig
)

_configs = {
    "unitree_aliengo_walk": UnitreeAliengoEnvConfig
}


def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]
