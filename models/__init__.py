"""ML model subsystem for the quant lab.

``LSTMModel`` is resolved lazily because it imports torch, which is optional
outside the GPU container. It was previously absent from this module's exports
entirely, so ``from models import LSTMModel`` failed even where torch WAS
installed.
"""

from typing import TYPE_CHECKING

from models.base import TASKS, BaseModel
from models.random_forest_model import RandomForestModel
from models.registry import ModelRegistry
from models.trainer import ModelTrainer
from models.xgboost_model import XGBoostModel

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from models.lstm_model import LSTMModel  # noqa: F401

__all__ = [
    "BaseModel",
    "TASKS",
    "XGBoostModel",
    "RandomForestModel",
    "LSTMModel",
    "ModelRegistry",
    "ModelTrainer",
]


def __getattr__(name: str):
    """Import :class:`LSTMModel` on first access (PEP 562)."""
    if name == "LSTMModel":
        from models.lstm_model import LSTMModel

        globals()[name] = LSTMModel
        return LSTMModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
