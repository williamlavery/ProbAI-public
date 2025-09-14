#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CachedLoader utilities: deterministic ordering, pair shuffling, and a light
in-memory batcher for paired tensors/sequences.

Version: 1.2.0
Date: 2025-08-27
Author: W. Lavery 
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• make_order(n, seed, epoch): deterministic permutation helper.
• FixedOrderSampler: torch Sampler that yields a fixed, predefined order.
• CachedLoader: minimal dataset-like iterable that batches two aligned sources
  (x0, x1). Supports:
    - Standard shuffling that *preserves pairing* (i == i),
    - Independent shuffling of x0 and x1 to *misalign pairings* (cross mode),
    - Immediate, in-memory misalignment when `shuffle_pairs=True`,
    - Optional device transfer and non_blocking copies,
    - drop_last semantics consistent with PyTorch DataLoader.

Design notes
------------
- Works with either tensors (fast indexing) or indexable sequences (falls back
  to stacking gathered items).
- Deterministic behavior comes from a local torch.Generator seeded with seed+epoch.
"""

import torch
from torch.utils.data import Sampler
from typing import Sequence, Optional


def make_order(n: int, seed: int = 0, epoch: int = 0) -> list[int]:
    """
    Create a deterministic permutation of range(n) using a local RNG.
    """
    g = torch.Generator().manual_seed(int(seed) + int(epoch))
    return torch.randperm(n, generator=g).tolist()


class FixedOrderSampler(Sampler):
    """
    Sampler that yields a fixed, user-provided order of indices.
    """
    def __init__(self, order: Sequence[int]):
        self.order = list(map(int, order))

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


class CachedLoader:
    """
    Minimal iterable that batches two aligned sources (x0, x1).

    Behavior
    --------
    - In *paired mode* (default), a single `order` is used for both x0 and x1,
      preserving i == i pairing.
    - In *cross-shuffled mode* (via `shuffle_pairings()`), x0 and x1 get
      *independent* orders, deliberately misaligning pairs.
    - If `shuffle_pairs=True` at construction, x0 and x1 are misaligned
      **immediately in memory** (not just by iteration order).

    Parameters
    ----------
    x0, x1 : Tensor or indexable sequence
        Sources with the same length.
    batch_size : int, default 32
    order : Optional[Sequence[int]]
        Initial order for paired mode. If None, uses range(len(x0)).
    drop_last : bool, default False
    device : Optional[str or torch.device], default None
    non_blocking : bool, default False
    shuffle_pairs : bool, default False
        If True, independently permute x0 and x1 in-memory right away.
    seed : Optional[int], default 0
        Seed used for deterministic shuffles (including `shuffle_pairs`).
    epoch : int, default 0
        Epoch offset added to the seed.
    """
    def __init__(self,
                 x0,
                 x1,
                 batch_size: int = 32,
                 order: Optional[Sequence[int]] = None,
                 drop_last: bool = False,
                 device=None,
                 non_blocking: bool = False,
                 shuffle_pairs: bool = False,
                 seed: Optional[int] = 0,
                 epoch: int = 0):
        assert len(x0) == len(x1), "x0 and x1 must have the same length."
        self.x0, self.x1 = x0, x1
        self.bs = int(batch_size)
        self.drop_last = bool(drop_last)
        self.device = device
        self.nb = bool(non_blocking)

        # Orders for cross-shuffled iteration (None means paired mode)
        self.order0: Optional[list[int]] = None
        self.order1: Optional[list[int]] = None

        # Optionally misalign pairs IMMEDIATELY in memory.
        if shuffle_pairs:
            self._inplace_misalign(seed=seed, epoch=epoch)
        self.x0_stacked = torch.stack(self.x0, axis=0) 
        self.x1_stacked = torch.stack(self.x1, axis=0)
        # Default (paired) ordering for iteration
        self.set_order(order)

    # ----------------------- helpers -----------------------

    @staticmethod
    def _permute_data(src, perm: Sequence[int]):
        """Return src permuted by perm, preserving tensor vs. sequence type."""
        if isinstance(src, torch.Tensor):
            return src[perm]
        # For generic sequences, return a list in permuted order
        return [src[i] for i in perm]

    def _inplace_misalign(self, seed: Optional[int] = 0, epoch: int = 0) -> None:
        """
        Independently permute x0 and x1 in memory (one-time), so that subsequent
        paired iteration uses misaligned pairs by construction.
        """
        n = len(self.x0)
        if seed is None:
            perm0 = torch.randperm(n).tolist()
            perm1 = torch.randperm(n).tolist()
        else:
            base = int(seed) + 2 * int(epoch)
            g0 = torch.Generator().manual_seed(base)
            g1 = torch.Generator().manual_seed(base + 1)
            perm0 = torch.randperm(n, generator=g0).tolist()
            perm1 = torch.randperm(n, generator=g1).tolist()

        self.x0 = self._permute_data(self.x0, perm0)
        self.x1 = self._permute_data(self.x1, perm1)

        # Ensure iteration starts in paired mode over [0..n-1]
        self.order0 = None
        self.order1 = None

    # ---------------------- public API ----------------------

    def set_order(self, order: Optional[Sequence[int]] = None) -> None:
        """Set a single shared order (paired mode)."""
        n = len(self.x0)
        self.order = list(range(n)) if order is None else list(map(int, order))
        total = len(self.order)
        if self.drop_last:
            self._num_batches = (total // self.bs)
        else:
            self._num_batches = (total + self.bs - 1) // self.bs

    def shuffle(self, seed: Optional[int] = None, epoch: int = 0) -> None:
        """Standard shuffle: shuffle pairs together (keeps i==i pairing)."""
        n = len(self.x0)
        if seed is None:
            order = torch.randperm(n).tolist()
        else:
            g = torch.Generator().manual_seed(int(seed) + int(epoch))
            order = torch.randperm(n, generator=g).tolist()
        self.set_order(order)

    def shuffle_pairings(self, seed: Optional[int] = None, epoch: int = 0) -> None:
        """
        Independently shuffle x0 and x1 orders for iteration (cross-shuffled mode).
        Does NOT modify data in memory; only affects iteration order.
        """
        n = len(self.x0)
        if seed is None:
            self.order0 = torch.randperm(n).tolist()
            self.order1 = torch.randperm(n).tolist()
        else:
            base = int(seed) + 2 * int(epoch)
            g0 = torch.Generator().manual_seed(base)
            g1 = torch.Generator().manual_seed(base + 1)
            self.order0 = torch.randperm(n, generator=g0).tolist()
            self.order1 = torch.randperm(n, generator=g1).tolist()

        # Keep length/batch math consistent
        total = n
        if self.drop_last:
            self._num_batches = (total // self.bs)
        else:
            self._num_batches = (total + self.bs - 1) // self.bs

    def __len__(self) -> int:
        """Number of batches yielded according to current order and drop_last."""
        return self._num_batches

    def _gather(self, src, idxs):
        """
        Gather a batch from `src` at positions `idxs`.

        If `src` is a tensor, uses fancy indexing. Otherwise, stacks items
        retrieved with Python indexing. Optionally moves the batch to `self.device`.
        """
        b = src[idxs] if isinstance(src, torch.Tensor) else torch.stack([src[i] for i in idxs], 0)
        return b.to(self.device, non_blocking=self.nb) if self.device is not None else b

    def __iter__(self):
        """
        Yield batches (x0_batch, x1_batch) according to the current mode:

        - Cross-shuffled mode: when both `order0` and `order1` are set.
        - Paired mode: otherwise, use the shared `order`.
        """
        bs = self.bs

        # Cross-shuffled pairing mode
        if self.order0 is not None and self.order1 is not None:
            limit = (len(self.order0) // bs) * bs if self.drop_last else len(self.order0)
            for s in range(0, limit, bs):
                idxs0 = self.order0[s:s+bs]
                idxs1 = self.order1[s:s+bs]
                if self.drop_last and (len(idxs0) < bs or len(idxs1) < bs):
                    break
                yield self._gather(self.x0, idxs0), self._gather(self.x1, idxs1)
            return

        # Paired mode
        order = self.order
        limit = (len(order) // bs) * bs if self.drop_last else len(order)
        for s in range(0, limit, bs):
            idxs = order[s:s+bs]
            if self.drop_last and len(idxs) < bs:
                break
            yield self._gather(self.x0, idxs), self._gather(self.x1, idxs)
