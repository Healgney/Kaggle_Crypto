from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

class LoggerCallback(Callback):
    def __init__(self, batch_interval: int) -> None:
        super().__init__()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int
    ) -> None:
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
