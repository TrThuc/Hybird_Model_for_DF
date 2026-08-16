from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

from torch.utils.data import Sampler


class ExactSourceBalancedSampler(Sampler[int]):
    """
    Lấy chính xác:
      - real_target frame từ source real;
      - fake_target frame từ các source fake;
      - fake_target được chia gần đều cho từng loại fake.

    Train:
      shuffle=True và thay đổi mẫu theo epoch bằng set_epoch(epoch).

    Validation:
      shuffle=False để giữ một tập con cố định và tái lập được.
    """

    def __init__(
        self,
        records: Sequence[Dict],
        real_source: str,
        fake_sources: Sequence[str],
        real_target: int,
        fake_target: int,
        seed: int = 5,
        shuffle: bool = True,
    ) -> None:
        self.records = records
        self.real_source = str(real_source)
        self.fake_sources = [str(source) for source in fake_sources]
        self.real_target = int(real_target)
        self.fake_target = int(fake_target)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        if self.real_target <= 0 or self.fake_target <= 0:
            raise ValueError(
                "real_target và fake_target phải lớn hơn 0."
            )

        if not self.fake_sources:
            raise ValueError("fake_sources không được rỗng.")

        grouped = defaultdict(list)

        for index, record in enumerate(records):
            source = str(record.get("source", ""))
            grouped[source].append(index)

        self.real_indices = grouped[self.real_source]
        self.fake_indices = {
            source: grouped[source]
            for source in self.fake_sources
        }

        if len(self.real_indices) < self.real_target:
            raise ValueError(
                f"Source real {self.real_source!r} chỉ có "
                f"{len(self.real_indices)} frame, nhưng yêu cầu "
                f"{self.real_target}."
            )

        quotas = self._fixed_fake_quotas()

        for source, quota in quotas.items():
            available = len(self.fake_indices[source])
            if available < quota:
                raise ValueError(
                    f"Source fake {source!r} chỉ có {available} frame, "
                    f"nhưng cần ít nhất {quota}."
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _fixed_fake_quotas(self) -> Dict[str, int]:
        base = self.fake_target // len(self.fake_sources)
        remainder = self.fake_target % len(self.fake_sources)

        return {
            source: base + (1 if index < remainder else 0)
            for index, source in enumerate(self.fake_sources)
        }

    def _epoch_fake_quotas(
        self,
        rng: random.Random,
    ) -> Dict[str, int]:
        base = self.fake_target // len(self.fake_sources)
        remainder = self.fake_target % len(self.fake_sources)

        source_order = list(self.fake_sources)

        # Luân phiên loại fake nhận phần dư giữa các epoch.
        if self.shuffle:
            rng.shuffle(source_order)

        quotas = {source: base for source in self.fake_sources}

        for source in source_order[:remainder]:
            quotas[source] += 1

        return quotas

    @staticmethod
    def _sample_without_replacement(
        population: Sequence[int],
        count: int,
        rng: random.Random,
    ) -> List[int]:
        if count > len(population):
            raise ValueError(
                f"Không đủ phần tử: cần {count}, có {len(population)}."
            )

        if count == len(population):
            result = list(population)
            rng.shuffle(result)
            return result

        return rng.sample(list(population), count)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)

        selected = self._sample_without_replacement(
            self.real_indices,
            self.real_target,
            rng,
        )

        fake_quotas = self._epoch_fake_quotas(rng)

        for source in self.fake_sources:
            selected.extend(
                self._sample_without_replacement(
                    self.fake_indices[source],
                    fake_quotas[source],
                    rng,
                )
            )

        if self.shuffle:
            rng.shuffle(selected)

        return iter(selected)

    def __len__(self) -> int:
        return self.real_target + self.fake_target

    def summary(self) -> Dict[str, int]:
        result = {
            self.real_source: self.real_target,
        }
        result.update(self._fixed_fake_quotas())
        return result
