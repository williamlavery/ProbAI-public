#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProbAIMnistDataset: paired (x0, x1) MNIST dataset for flow/denoising training.

Version: 1.1.0
Date: 2025-08-26
Author: ProbAI 2025 (edited by W. Lavery) 
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• A thin wrapper around a torchvision MNIST-like dataset that yields *pairs*
  (x0, x1) suitable for flow-matching or denoising-style training.
• x1 is a standard MNIST image; x0 is either:
    - a noisy variant (Gaussian noise) with *deterministic per-item RNG*, or
    - a copy of an MNIST image with a deterministic binary square “crop noise”
      patch stamped onto it.
• Optional per-epoch pair reshuffling to vary (x0, x1) pairings while keeping
  per-item determinism for reproducibility.

------------------------------------------------------------------------------
Usage sketch
------------------------------------------------------------------------------
from torchvision import datasets, transforms
mnist = datasets.MNIST(root=".", train=True, download=True,
                       transform=transforms.ToTensor())

pairs = ProbAIMnistDataset(mnist, crop_noise=False, shuffle_pairs=True, shuffle_seed=0)
x0, x1 = pairs[0]  # x0: noise (or patched image), x1: MNIST image (both [1,H,W])

# Between epochs (optional):
# pairs.reshuffle_pairs(seed=epoch)

"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class ProbAIMnistDataset(Dataset):
    """
    Paired MNIST dataset producing tuples (x0, x1) with deterministic randomness.

    Behavior
    --------
    • `x1` is taken directly from the wrapped MNIST-like dataset (index `idx`).
    • If `crop_noise=True`:
        - Choose pairing index `j` (optionally shuffled).
        - Take `x0` as a clone of image `j` and *stamp* a square patch of ones
          of size `crop_size × crop_size` at a deterministic location computed
          from a per-item seed.
    • Else (`crop_noise=False`):
        - `x0` is a deterministic Gaussian noise tensor generated from a
          per-item seed, with shape [1, H, W].

    Determinism & shuffling
    -----------------------
    - A stable, per-item seed is constructed from `base_seed` and `idx`.
    - `shuffle_pairs=True` introduces a fixed permutation (seeded) that maps
      each `idx` → `j` when forming the optional crop-noise pairing.
    - You may call `reshuffle_pairs(seed)` between epochs to change the pairing
      permutation while preserving deterministic per-item RNG.

    Parameters
    ----------
    mnist_dataset : torch.utils.data.Dataset
        Dataset whose __getitem__(idx)[0] returns a tensor image `[1, H, W]`
        (e.g., torchvision.datasets.MNIST with transforms.ToTensor()).
    crop_noise : bool, default False
        If True, `x0` is formed by stamping a binary square into another image;
        otherwise `x0` is Gaussian noise.
    crop_size : int, default 12
        Side length of the square stamped when `crop_noise=True`.
        Can remain unchanged, but feel free to experiment with this parameter.
    shuffle_pairs : bool, default False
        If True, use a permuted index `j` for the `x0` image when cropping.
    shuffle_seed : int, default 0
        Seed for the initial pairing permutation (when `shuffle_pairs=True`).

    Notes
    -----
    • This wrapper assumes grayscale images with shape [1, H, W] and square H==W.
    • All random draws use local `torch.Generator` instances to avoid perturbing
      global RNG state (important for reproducibility across dataloaders/workers).
    """

    def __init__(self, 
                 mnist_dataset, 
                 crop_noise: bool = False, 
                 crop_size: int = 12,
                 shuffle_pairs: bool = False,
                 shuffle_seed: int = 0):
        self.mnist_dataset = mnist_dataset
        self.crop_noise = crop_noise
        self.shuffle_pairs = shuffle_pairs
        self.base_seed = shuffle_seed

        if shuffle_pairs:
            torch.manual_seed(shuffle_seed)  # For reproducibility
            self.shuffle_idx = torch.randperm(len(mnist_dataset))
            
        # Can remain unchanged, but feel free to experiment with this parameter
        self.crop_size = crop_size
        
        self.img_dim0 = mnist_dataset.data.shape[1]
        self.img_dim1 = mnist_dataset.data.shape[2]

    def __len__(self) -> int:
        """Number of items in the underlying dataset."""
        return self.mnist_dataset.__len__()

    def __getitem__(self, idx: int, seed: int | None = None):
        """
        Return a deterministic pair (x0, x1) for item `idx`.

        Parameters
        ----------
        idx : int
            Sample index.
        seed : Optional[int]
            Optional override seed for this retrieval; if None, uses `base_seed`.

        Returns
        -------
        (x0, x1) : Tuple[torch.Tensor, torch.Tensor]
            Tensors of shape [1, H, W], dtype inherited from the underlying data.
        """
        # Build a per-item seed that's stable across runs and workers
        base = self.base_seed if seed is None else seed
        per_item_seed = int((base + idx) % (2**63 - 1))

        # x1 is deterministic because it's indexed, not randomly generated
        x1 = self.mnist_dataset.__getitem__(idx)[0].detach()

        if self.crop_noise:
            # Optional deterministic pairing
            j = self.shuffle_idx[idx] if getattr(self, "shuffle_pairs", False) else idx
            x0 = self.mnist_dataset.__getitem__(j)[0].detach().clone()

            # Use local generators so we don't affect global RNG state
            gen0 = torch.Generator(device=x0.device).manual_seed(per_item_seed)
            idx0 = torch.randint(
                0, self.img_dim0 - self.crop_size, (1,), generator=gen0
            ).item()

            gen1 = torch.Generator(device=x0.device).manual_seed(per_item_seed + 1)
            idx1 = torch.randint(
                0, self.img_dim1 - self.crop_size, (1,), generator=gen1
            ).item()

            x0[0, idx0:idx0 + self.crop_size, idx1:idx1 + self.crop_size] = 1
        else:
            # Deterministic Gaussian noise per item
            gen = torch.Generator(device=x1.device).manual_seed(per_item_seed)
            x0 = torch.randn(1, self.img_dim0, self.img_dim0, generator=gen)
    
        return x0, x1

    def reshuffle_pairs(self, seed: int | None = None) -> None:
        """
        Regenerate the pairing permutation used when `shuffle_pairs=True`.

        Call between epochs to rearrange pairs without changing per-item RNG.
        """
        # You can call this between epochs to rearange the pairs
        # Is probably not that influential/important given the size of the dataset and how few epochs we need
        if self.shuffle_pairs:
            if seed is not None:
                torch.manual_seed(seed)
            self.shuffle_idx = torch.randperm(len(self.mnist_dataset))
