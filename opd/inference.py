"""Replicated inference workers: no DDP, optimizer, gradient or ZeRO wrapping.

Torchrun supervises failures; Gloo collects Python records on rank zero so
nonlinear metrics are computed globally, never averaged across rank summaries.
"""

import contextlib
import os
from datetime import timedelta

import torch
import torch.distributed as dist


class InferenceWorkers:
    def __init__(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    @property
    def main(self):
        return self.rank == 0

    def shard(self, rows):
        return rows[self.rank::self.world_size]

    def gather(self, records):
        if self.world_size == 1:
            return records
        gathered = [None] * self.world_size if self.main else None
        dist.gather_object(records, gathered, dst=0)
        return [record for part in gathered for record in part] if self.main else None


@contextlib.contextmanager
def inference_workers():
    created = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        cpu = os.environ.get("ACCELERATE_USE_CPU", "").lower() in ("1", "true", "yes")
        if torch.cuda.is_available() and not cpu:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
        dist.init_process_group("gloo", timeout=timedelta(hours=2))
        created = True
    try:
        yield InferenceWorkers()
    finally:
        if created:
            dist.destroy_process_group()
