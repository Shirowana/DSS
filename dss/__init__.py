from peft.utils import register_peft_method

from .config import DSSConfig
from .layer import DSSLayer, DSSLinear
from .model import DSSModel

__all__ = [
    "DSSConfig",
    "DSSLayer",
    "DSSLinear",
    "DSSModel",
]

try:
    register_peft_method(name="dss", model_cls=DSSModel, config_cls=DSSConfig)
except KeyError as exc:
    if "already PEFT method" not in str(exc):
        raise
