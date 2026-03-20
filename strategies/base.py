"""Abstract base class for all trading strategies.

Every strategy in the quant-lab must subclass :class:`BaseStrategy` and
implement the three abstract methods:

* ``generate_signals`` -- produce a buy / sell / hold signal series.
* ``get_parameters``   -- return the strategy's tuneable hyper-parameters.
* ``get_dependencies``  -- declare which features or indicators the strategy
  requires so the data pipeline can prepare them ahead of time.
"""

from abc import ABC, abstractmethod
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class that all trading strategies must inherit from.

    Subclasses are expected to set the ``name`` and ``description`` class
    attributes and to implement the abstract interface defined below.

    Attributes:
        name: A short, unique identifier for the strategy
            (e.g. ``"ema_crossover"``).
        description: A human-readable explanation of what the strategy does.
    """

    name: str
    description: str

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Produce trading signals from market data.

        Args:
            df: A DataFrame that **must** contain at least a ``close``
                column.  Depending on the strategy, additional columns
                (e.g. ``volume``, pre-computed indicators) may be required
                -- see :meth:`get_dependencies`.

        Returns:
            A :class:`pandas.Series` aligned with *df*'s index where each
            value is one of:

            *  ``1``  -- **buy** signal
            * ``-1``  -- **sell** signal
            *  ``0``  -- **hold** (no action)
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return the strategy's current parameter set.

        Returns:
            A JSON-serialisable dictionary mapping parameter names to
            their values (e.g. ``{"fast_period": 12, "slow_period": 26}``).
        """
        ...

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Declare the features / indicators that this strategy needs.

        The data pipeline uses this list to ensure all required columns
        are present in the DataFrame before ``generate_signals`` is
        called.

        Returns:
            A list of column names that **must** exist in the input
            DataFrame (e.g. ``["close", "volume"]``).
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params = self.get_parameters()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Check that *df* contains every column listed in dependencies.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing = [col for col in self.get_dependencies() if col not in df.columns]
        if missing:
            raise ValueError(
                f"Strategy '{self.name}' requires columns {missing} "
                f"which are missing from the input DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        logger.debug(
            "DataFrame validated for strategy '%s' -- all dependencies present.",
            self.name,
        )
