"""LSTM classifier for binary next-bar direction prediction (1 = up, 0 = down).

Expects 3-D input arrays of shape ``(n_samples, sequence_length, n_features)``
produced from the Hive-partitioned feature store.  Use
:meth:`~financials.etl_pipeline.load.reader.MLDataLoader.load_pandas` to pull
a symbol's full history, add the next-bar direction target, and then build
sliding windows -- see ``notebooks/06_lstm_training.ipynb`` for the full
training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import BaseModel

logger = logging.getLogger(__name__)

_Array = Union[np.ndarray, pd.DataFrame, pd.Series]


# ---------------------------------------------------------------------------
# Internal network
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    """Stacked LSTM with a single linear head for binary classification."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # nn.LSTM only applies dropout *between* layers, not on the output
            # of the final layer, so we pair it with an explicit Dropout below.
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)               # (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])   # last-timestep representation
        return self.fc(out).squeeze(-1)     # (batch,) logits


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class LSTMModel(BaseModel):
    """LSTM classifier for next-bar price direction.

    Parameters
    ----------
    name:
        Unique identifier used by the model registry.
    n_features:
        Number of input features per timestep (columns in your feature window).
    sequence_length:
        Number of past bars fed into the LSTM at inference time.
    hidden_size:
        LSTM hidden state dimensionality.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied after the final LSTM layer and between
        intermediate layers (when ``num_layers > 1``).
    learning_rate:
        Adam learning rate.
    batch_size:
        Mini-batch size for training and inference.
    epochs:
        Number of full passes over the training data.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.
    random_state:
        Seed for PyTorch and NumPy reproducibility.
    """

    model_type: str = "lstm"

    def __init__(
        self,
        name: str,
        n_features: int = 29,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.name = name
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.train_history: list[dict[str, float]] = []

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(random_state)
        self._net = _LSTMNet(n_features, hidden_size, num_layers, dropout).to(self.device)
        logger.info("Initialised LSTMModel %r on %s", name, self.device)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: _Array, y: _Array, *, val_split: float = 0.1) -> None:
        """Fit the LSTM on windowed feature sequences.

        Parameters
        ----------
        X:
            3-D ``float32`` array of shape ``(n_samples, sequence_length,
            n_features)``.  A ``pd.DataFrame`` is rejected -- pass a NumPy
            array built from sliding windows.
        y:
            1-D binary labels (0 or 1), one per sample.
        val_split:
            Fraction of the **training** data held out for per-epoch
            validation reporting.  This split is taken from the *end* of the
            array so it remains time-ordered.
        """
        X_arr = _to_numpy_3d(X)
        y_arr = _to_numpy_1d(y)

        split = int(len(X_arr) * (1 - val_split))
        X_tr, X_val = X_arr[:split], X_arr[split:]
        y_tr, y_val = y_arr[:split], y_arr[split:]

        train_loader = _make_loader(X_tr, y_tr, self.batch_size, shuffle=True)
        val_loader = _make_loader(X_val, y_val, self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = _run_epoch(
                self._net, train_loader, criterion, optimizer, self.device, training=True
            )
            val_loss, val_acc = _run_eval(self._net, val_loader, criterion, self.device)
            scheduler.step(val_loss)

            self.train_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "[%s] epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                    self.name,
                    epoch,
                    self.epochs,
                    train_loss,
                    val_loss,
                    val_acc,
                )

    def predict(self, X: _Array) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: _Array) -> np.ndarray:
        """Return class-probability estimates of shape ``(n_samples, 2)``."""
        X_arr = _to_numpy_3d(X)
        self._net.eval()

        dummy_y = np.zeros(len(X_arr), dtype="float32")
        loader = _make_loader(X_arr, dummy_y, self.batch_size, shuffle=False)

        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, _ in loader:
                chunks.append(self._net(xb.to(self.device)).cpu())

        logits = torch.cat(chunks).numpy()
        proba_pos = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "random_state": self.random_state,
        }

    def save(self, path: str) -> None:
        """Persist the model weights and hyperparameters to *path* (.pt)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "hyperparameters": self.get_hyperparameters(),
                "state_dict": self._net.state_dict(),
                "train_history": self.train_history,
            },
            path,
        )
        logger.info("Saved LSTMModel %r to %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """Load a previously saved model from *path* and return an instance."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(name=checkpoint["name"], **checkpoint["hyperparameters"])
        instance._net.load_state_dict(checkpoint["state_dict"])
        instance.train_history = checkpoint.get("train_history", [])
        logger.info("Loaded LSTMModel from %s", path)
        return instance


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_numpy_3d(X: _Array) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype="float32")
    else:
        arr = np.asarray(X, dtype="float32")
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (n_samples, seq_len, n_features), got shape {arr.shape}. "
            "Build sliding windows before calling train() / predict()."
        )
    return arr


def _to_numpy_1d(y: _Array) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.to_numpy(dtype="float32").ravel()
    return np.asarray(y, dtype="float32").ravel()


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False)


def _run_epoch(
    net: _LSTMNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    training: bool,
) -> float:
    net.train() if training else net.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


def _run_eval(
    net: _LSTMNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    net.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
    n = len(loader.dataset)  # type: ignore[arg-type]
    return total_loss / n, correct / n
