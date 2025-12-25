import argparse
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.loss import WaveletLoss
from model.mlp import mlp
from model.wed import WED
from utils.utils import (
    M1_HEADERS,
    M2_HEADERS,
    M3_HEADERS,
    M4_HEADERS,
    M5_HEADERS,
    MOTHERDIR,
    OFLESDataset,
    R2Score,
    trainDataCollecter,
)


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"


@dataclass(frozen=True)
class TrainingConfig:
    Re: str = "R4"
    Mconf: str = "3"
    model_mode: str = "WED"
    wavelet: bool = False
    train_fraction: float = 1.0
    train_rows: int | None = None
    split_size: float = 0.8
    epochs: int = 5
    learning_rate: float = 0.001
    patience: int = 60
    seed: int = 42

    @classmethod
    def from_args(cls) -> "TrainingConfig":
        parser = argparse.ArgumentParser(description="Train with controllable dataset size.")
        parser.add_argument("--Re", default=cls.Re, help="Reynolds number selector, e.g., R3, R4, R53")
        parser.add_argument("--Mconf", default=cls.Mconf, help="Model configuration, matches M*_HEADERS")
        parser.add_argument("--model-mode", choices=["WED", "MLP"], default=cls.model_mode)
        parser.add_argument("--wavelet", action="store_true", help="Use WaveletLoss instead of MSE")
        parser.add_argument("--train-fraction", type=float, default=cls.train_fraction, help="Fraction of train set to use (0-1].")
        parser.add_argument(
            "--train-rows",
            type=int,
            default=cls.train_rows,
            help="Exact number of train rows to use (overrides fraction).",
        )
        parser.add_argument(
            "--split-size",
            type=float,
            default=cls.split_size,
            help="Portion of selected data kept for training (rest for val).",
        )
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--patience", type=int, default=cls.patience)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parsed = parser.parse_args()
        return cls(
            Re=parsed.Re,
            Mconf=str(parsed.Mconf),
            model_mode=parsed.model_mode,
            wavelet=parsed.wavelet,
            train_fraction=parsed.train_fraction,
            train_rows=parsed.train_rows,
            split_size=parsed.split_size,
            epochs=parsed.epochs,
            learning_rate=parsed.learning_rate,
            patience=parsed.patience,
            seed=parsed.seed,
        )

    @property
    def group_name(self) -> str:
        suffix = f"R10{self.Re[1]}"
        return f"wed_{suffix}" if self.model_mode == "WED" else f"mlp_{suffix}"

    @property
    def dataset_tag(self) -> str:
        return f"M{self.Mconf}_wavelet" if self.wavelet else f"M{self.Mconf}"


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    checkpoints: Path
    logs: Path
    traced: Path
    best_model: Path
    log_file: Path
    history_file: Path

    @classmethod
    def create(cls, cfg: TrainingConfig) -> "OutputLayout":
        root = MOTHERDIR / "runs" / cfg.group_name
        checkpoints = root / "checkpoints"
        logs = root / "logs"
        traced = root / "traced"
        best_model = checkpoints / f"{cfg.group_name}_model_{cfg.dataset_tag}.pt"
        log_file = logs / f"{cfg.group_name}_training_log_{cfg.dataset_tag}.txt"
        history_file = logs / f"{cfg.group_name}_training_history_{cfg.dataset_tag}.json"
        for path in (checkpoints, logs, traced):
            path.mkdir(parents=True, exist_ok=True)
        return cls(root, checkpoints, logs, traced, best_model, log_file, history_file)


class HistoryBuffer:
    def __init__(self) -> None:
        self._store: MutableMapping[str, list[float]] = {}

    def append(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self._store.setdefault(key, []).append(float(value))

    def serializable(self) -> Mapping[str, Sequence[float]]:
        return dict(self._store)


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def _select_headers(name: str) -> Sequence[str]:
    headers = {
        "1": M1_HEADERS,
        "2": M2_HEADERS,
        "3": M3_HEADERS,
        "4": M4_HEADERS,
        "5": M5_HEADERS,
    }
    return headers[str(name)]


def _maybe_subsample(df, fraction: float, rows: int | None, seed: int):
    if rows is not None:
        return df.sample(n=min(rows, len(df)), random_state=seed, replace=False).reset_index(drop=True)
    if fraction < 1.0:
        return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return df


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.outputs = OutputLayout.create(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = HistoryBuffer()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model().to(self.device).double()
        self.criterion = WaveletLoss(wavelet="db1") if cfg.wavelet else nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.2)

    def _build_dataloaders(self):
        _, train_norm, _, _ = trainDataCollecter(self.cfg.Re)
        headers = _select_headers(self.cfg.Mconf)
        dt = _maybe_subsample(train_norm.filter(headers, axis=1), self.cfg.train_fraction, self.cfg.train_rows, self.cfg.seed)

        mask = np.random.rand(len(dt)) < self.cfg.split_size
        train = dt[mask].reset_index(drop=True)
        val = dt[~mask].reset_index(drop=True)

        train_dataset = OFLESDataset(train)
        val_dataset = OFLESDataset(val)

        batch_sz_trn = 4096
        batch_sz_val = batch_sz_trn // 4
        return (
            torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_sz_trn, shuffle=True),
            torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_sz_val, shuffle=False),
        )

    def _build_model(self) -> nn.Module:
        input_dim = self.train_loader.dataset[0].shape[0] - 1  # type: ignore[index]
        if self.cfg.model_mode == "WED":
            return WED(in_channels=input_dim, out_channels=1, bilinear=True)
        return mlp(input_size=input_dim, output_size=1, hidden_layers=5, neurons_per_layer=[60, 60, 60, 60, 60])

    def _forward_epoch(self, stage: Stage) -> tuple[float, float]:
        is_train = stage is Stage.TRAIN
        loader = self.train_loader if is_train else self.val_loader
        self.model.train() if is_train else self.model.eval()
        running_loss = 0.0
        y_true: list[torch.Tensor] = []
        y_pred: list[torch.Tensor] = []
        iterator: Iterable = tqdm(loader, desc=f"{stage.value.upper()} | {self.cfg.group_name}", leave=False)

        grad_context = torch.enable_grad if is_train else torch.no_grad
        with grad_context():
            for batch in iterator:
                inputs = batch[:, 0:-1].to(self.device)
                target = batch[:, -1].to(self.device)
                if is_train:
                    self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), target)
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item()
                y_true.append(target)
                y_pred.append(outputs.squeeze())
                iterator.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(loader)
        coefficient = R2Score(torch.cat(y_true), torch.cat(y_pred)).item()
        return epoch_loss, coefficient

    def _log_epoch(self, epoch: int, metrics: Mapping[str, float], duration: float) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        line = (
            f"Epoch [{epoch}/{self.cfg.epochs}], "
            f"Train Loss: {metrics['train_loss']:.4f}, Val Loss: {metrics['val_loss']:.4f}, "
            f"Train R^2: {metrics['train_r2']:.4f}, Val R^2: {metrics['val_r2']:.4f}, "
            f"LR: {lr:.6f}, Time: {duration:.2f}s"
        )
        logging.info(line)
        with self.outputs.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _trace_model(self) -> None:
        data_iter = iter(self.train_loader)
        scripted = torch.jit.trace(self.model, next(data_iter)[:, 0:-1].to(self.device))
        scripted.save(self.outputs.traced / f"{self.cfg.group_name}_traced_model_{self.cfg.dataset_tag}.pt")

    def train(self) -> None:
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch_idx in range(1, self.cfg.epochs + 1):
            with timed() as elapsed:
                train_loss, train_r2 = self._forward_epoch(Stage.TRAIN)
                val_loss, val_r2 = self._forward_epoch(Stage.VAL)
                self.scheduler.step()

            self.history.append(
                train_loss=train_loss,
                val_loss=val_loss,
                train_coefficient=train_r2,
                val_coefficient=val_r2,
                learning_rates=self.optimizer.param_groups[0]["lr"],
                epoch_times=elapsed(),
            )
            self._log_epoch(epoch_idx, {"train_loss": train_loss, "val_loss": val_loss, "train_r2": train_r2, "val_r2": val_r2}, elapsed())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.outputs.best_model)
                logging.info("New optimum detected; checkpoint persisted.")
            else:
                patience_counter += 1
                logging.info("No improvement observed; patience incremented.")

            if patience_counter >= self.cfg.patience:
                logging.info("Early stopping triggered; exiting training loop.")
                break

        with self.outputs.history_file.open("w", encoding="utf-8") as fh:
            json.dump(self.history.serializable(), fh)
        self._trace_model()


def main() -> None:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
    cfg = TrainingConfig.from_args()
    _seed_everything(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()
    logging.info("Training complete. Artifacts written to %s", trainer.outputs.root)


if __name__ == "__main__":
    main()
