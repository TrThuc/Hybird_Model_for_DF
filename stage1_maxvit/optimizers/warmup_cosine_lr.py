import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """Epoch-level linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        n_epoch: int,
        warmup_epochs: int = 1,
        warmup_start_lr: float = 1e-4,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ) -> None:
        if n_epoch <= 0:
            raise ValueError("n_epoch must be positive")
        if warmup_epochs < 0 or warmup_epochs >= n_epoch:
            raise ValueError("warmup_epochs must satisfy 0 <= value < n_epoch")
        if warmup_start_lr < 0 or min_lr < 0:
            raise ValueError("warmup_start_lr and min_lr must be non-negative")

        peak_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if any(warmup_start_lr > peak_lr for peak_lr in peak_lrs):
            raise ValueError("warmup_start_lr cannot exceed the peak learning rate")
        if any(min_lr > peak_lr for peak_lr in peak_lrs):
            raise ValueError("min_lr cannot exceed the peak learning rate")

        self.n_epoch = int(n_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_start_lr = float(warmup_start_lr)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def _lr_for(self, peak_lr: float, epoch_index: int) -> float:
        epoch_index = max(0, int(epoch_index))
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return self.warmup_start_lr
            progress = epoch_index / (self.warmup_epochs - 1)
            return self.warmup_start_lr + progress * (
                peak_lr - self.warmup_start_lr
            )

        cosine_steps = self.n_epoch - self.warmup_epochs
        if cosine_steps <= 1:
            return self.min_lr
        progress = (epoch_index - self.warmup_epochs) / (cosine_steps - 1)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (peak_lr - self.min_lr) * cosine

    def get_lr(self):
        return [
            self._lr_for(base_lr, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def apply_current_lr(self) -> None:
        """Recompute optimizer LRs after changing/restoring base_lrs."""
        values = self.get_lr()
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value
        self._last_lr = values
