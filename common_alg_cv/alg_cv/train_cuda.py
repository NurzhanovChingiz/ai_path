"""CUDA-accelerated training functions."""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Dataloader need pin_memory=True, num_workers>0
# funy i get acceleration without pin_memory on rocm
# from 19 sec to 15 sec per epoch on mnist with vgg9


def move_to_device(
        batch: Any,
        device: torch.device,
        non_blocking: bool = True) -> Any:
    """Recursively move tensors in a batch (tensor, dict, list, tuple) to the given device."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking)
                for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking)
                           for v in batch)
    return batch  # leave non-tensors as-is


def _record_stream(obj: Any, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(obj):
        obj.record_stream(stream)
    elif isinstance(obj, dict):
        for v in obj.values():
            _record_stream(v, stream)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _record_stream(v, stream)


class CUDAPrefetcher:
    """Overlaps CPU→GPU copy of the *next* batch with GPU compute on the *current* batch.

    Requires:
      - DataLoader(..., pin_memory=True)
      - CUDA device
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        """Initialize the prefetcher with a DataLoader and CUDA device."""
        if device.type != "cuda":
            raise ValueError("CUDAPrefetcher requires a CUDA device.")
        self.base_loader = loader
        self.loader = iter(loader)
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        self._next = None
        self._exhausted = False
        self._preload()

    def _preload(self) -> None:
        if self._exhausted:
            self._next = None
            return
        try:
            batch = next(self.loader)
        except StopIteration:
            self._next = None
            self._exhausted = True
            return
        with torch.cuda.stream(self.copy_stream):
            self._next = move_to_device(batch, self.device, non_blocking=True)

    def __iter__(self) -> "CUDAPrefetcher":
        """Return self as the iterator."""
        return self

    def __next__(self) -> Any:
        """Return the next prefetched batch."""
        if self._next is None:
            raise StopIteration
        current = torch.cuda.current_stream(device=self.device)
        current.wait_stream(self.copy_stream)
        batch = self._next
        _record_stream(batch, current)
        self._preload()
        return batch

    # Backward-compatible alias with the original API
    def next(self) -> Any | None:
        """Return next batch or None when exhausted (alias for __next__)."""
        try:
            return self.__next__()
        except StopIteration:
            return None

    # ←— make it re-iterable per epoch
    def reset(self) -> None:
        """Rewind the underlying DataLoader iterator for the next epoch."""
        self.loader = iter(self.base_loader)
        self._exhausted = False
        self._next = None
        self._preload()


def train(
        model: nn.Module,
        dataloader: DataLoader[Any],
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        prefetcher: CUDAPrefetcher | None = None) -> None:
    """Train a model using CUDA prefetcher for overlapping data transfer and compute."""
    if dataloader.dataset is None:
        raise ValueError("dataloader.dataset cannot be None")
    size = len(dataloader.dataset)  # type: ignore[arg-type]

    model.train()
    if prefetcher is None:
        raise ValueError("CUDAPrefetcher is required for train()")
    batch = prefetcher.next()
    loss_train = 0
    count = 0
    while batch is not None:
        count += 1
        # unpack according to your dataset
        if isinstance(batch, dict):
            inputs = batch['inputs']
            targets = batch['targets']
        else:
            inputs, targets = batch  # (inputs, targets) tuple
        optimizer.zero_grad()
        pred = model(inputs)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        batch = prefetcher.next()
        loss_train += loss.item()

    if count > 0:
        print(
            f"Train count: {count:>7d}  [{size:>5d}/{size:>5d}], Avg loss: {(loss_train / count):>.8f}")
