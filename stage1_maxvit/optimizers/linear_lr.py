from torch.optim.lr_scheduler import LRScheduler


class LinearDecayLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        n_epoch,
        start_decay,
        last_epoch=-1,
    ):
        if n_epoch <= 0:
            raise ValueError("n_epoch must be positive")

        if start_decay < 0:
            raise ValueError("start_decay must be non-negative")

        if start_decay >= n_epoch:
            raise ValueError(
                "start_decay must be smaller than n_epoch"
            )

        self.n_epoch = int(n_epoch)
        self.start_decay = int(start_decay)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.start_decay:
            factor = 1.0
        else:
            decay_length = self.n_epoch - self.start_decay
            remaining = self.n_epoch - self.last_epoch
            factor = max(0.0, remaining / decay_length)

        return [
            base_lr * factor
            for base_lr in self.base_lrs
        ]