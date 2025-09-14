#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for dataset splits and training curves.

Version: 1.2.0
Date: 2025-08-26
Author: W. Lavery 
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• build_or_load_split(...)    : Deterministic train/val split with on-disk caching.
• plot_history                : Training curves for a single trainer (bug fixes).
• plot_histories              : Compare curves across multiple trainers.

Notes
-----
- Functions keep backwards-compatible signatures where possible.
- Commented-out code blocks are preserved exactly as provided.

"""

import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import DataLoader, Subset  # Subset used; DataLoader kept for compatibility
import time, os
from pathlib import Path
from typing import List, Dict, Any, Optional


def build_or_load_split(dataset, val_frac: float = 0.10, seed: int = 42,
                        split_dir: str | os.PathLike = "./data/splits",
                        name: str = "mnist_probai"):
    """
    Create (or load) a deterministic train/val split for a given dataset.

    Returns
    -------
    (train_ds, val_ds, train_idx, val_idx)

    Files created in `split_dir`:
      - {name}_split.pt           : dict with train_idx/val_idx and metadata
      - {name}_train_subset.pt    : dict with {"indices": train_idx, ...}
      - {name}_val_subset.pt      : dict with {"indices": val_idx, ...}
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    split_path = split_dir / f"{name}_split.pt"
    train_subset_path = split_dir / f"{name}_train_subset.pt"
    val_subset_path   = split_dir / f"{name}_val_subset.pt"

    n_total = len(dataset)

    def _valid(saved: dict) -> bool:
        if saved.get("n_total") != n_total:
            return False
        tr, va = saved.get("train_idx", []), saved.get("val_idx", [])
        if len(tr) + len(va) != n_total:
            return False
        return len(set(tr).intersection(va)) == 0

    def _save_indices(path, idx, kind):
        torch.save({
            "indices": idx,
            "kind": kind,
            "n_total": n_total,
            "name": name,
            "val_frac": val_frac,
            "seed": seed,
            "created_time": time.time(),
        }, path)

    # --- load or create split indices ---
    train_idx = val_idx = None
    if split_path.exists():
        try:
            saved = torch.load(split_path, map_location="cpu")
            if _valid(saved):
                train_idx = saved["train_idx"]
                val_idx   = saved["val_idx"]
            else:
                split_path.unlink(missing_ok=True)  # invalid; force remake
        except Exception:
            # Corrupted or incompatible file; remake
            split_path.unlink(missing_ok=True)

    if train_idx is None:
        # (re)create split
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n_total, generator=g).tolist()
        n_val = max(1, int(round(n_total * float(val_frac))))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        torch.save({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "n_total": n_total,
            "val_frac": float(val_frac),
            "seed": int(seed),
            "created_time": time.time(),
        }, split_path)

    # --- always (re)write subset index files to match the split ---
    _save_indices(train_subset_path, train_idx, "train")
    _save_indices(val_subset_path,   val_idx,   "val")

    # --- rehydrate Subset views for the provided dataset instance ---
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    return train_ds, val_ds, train_idx, val_idx

def plot_history(trainer, show_iter: bool = True, max_iters: Optional[int] = None, logy: bool = False):
    """
    Plot training curves.
    - If show_iter is True, also plots per-iteration loss (optionally truncated to the latest max_iters points).
    """
    # per-epoch
    plt.figure()
    plt.title("Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # x-axis is the recorded epoch indices (0-based internal)
    epochs_x = trainer.history.epochs if getattr(trainer.history, "epochs", None) else list(range(len(trainer.history.epoch_losses)))
    plt.plot(epochs_x, trainer.history.epoch_losses, marker="o", label="Train Loss")

    if any(v is not None for v in trainer.history.val_epoch_losses):
        val_x = [e for e, v in zip(trainer.history.epochs, trainer.history.val_epoch_losses) if v is not None]
        val_y = [v for v in trainer.history.val_epoch_losses if v is not None]
        # FIX: previously plotted val_y against implicit indices; now use val_x on x-axis
        plt.plot(val_x, val_y, marker="o", label="Val Loss", linestyle="--")

    plt.grid(True)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.show()

    # per-iteration (optional)
    if show_iter and getattr(trainer.history, "iters", None):
        iters = trainer.history.iters
        losses = trainer.history.iter_losses
        if max_iters is not None and max_iters > 0:
            iters = iters[-max_iters:]
            losses = losses[-max_iters:]
        plt.figure()
        plt.title("Iteration Loss")
        plt.xlabel("Global Iteration")
        plt.ylabel("Loss")
        plt.plot(iters, losses, linestyle="-", label="Train Loss")
        plt.grid(True)
        if logy:
            plt.yscale("log")
        plt.legend()
        plt.show()
        
def plot_histories(trainers, show_iter: bool = True, max_iters: Optional[int] = None,
                   logx: bool = False, xlim: float = None,
                   logy: bool = False, save_path_epoch: Optional[str] = None, save_path_iter: Optional[str] = None):
    """
    Plot training curves for one or more trainers.
    - If multiple trainers are given, each is plotted with a fixed color.
    - First trainer: red, second: pale blue, third: dark blue.
    - If show_iter is True, also plots per-iteration loss.
    """
    if not isinstance(trainers, (list, tuple)):
        trainers = [trainers]

    colors = ["red", "lightblue", "darkblue"]
    
    # ---- per-epoch ----
    plt.figure()
    # plt.title("Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss [a.u.]")

    for idx, trainer in enumerate(trainers):
        color = colors[idx % len(colors)]
        epochs_x = trainer.history.epochs if getattr(trainer.history, "epochs", None) else list(range(len(trainer.history.epoch_losses)))

        plt.plot(epochs_x, trainer.history.epoch_losses, 
                 marker="o", 
                 color=color, 
                 label=f"Train Loss C{idx+1}")
        
        if any(v is not None for v in trainer.history.val_epoch_losses):
            val_x = [e for e, v in zip(trainer.history.epochs, trainer.history.val_epoch_losses) if v is not None]
            val_y = [v for v in trainer.history.val_epoch_losses if v is not None]
            plt.plot(val_x, val_y, 
                     marker="s", 
                     linestyle=":", 
                     color=color, 
                     alpha=0.7, 
                     label=f"Val Loss C{idx+1}")
    
    plt.grid(True)
    if logy:
        plt.yscale("log")
    if logx:
        plt.xscale("log")
    if xlim is not None:
        plt.xlim(None, xlim)
    plt.legend()

    if save_path_epoch is not None:
        dirpath = os.path.dirname(save_path_epoch) or "."
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path_epoch)
        print(f"Saved epoch plot to {save_path_epoch}")
    plt.show()

    # ---- per-iteration (optional) ----
    if show_iter:
        plt.figure()
        #plt.title("Iteration Loss")
        plt.xlabel("Global Iteration")
        plt.ylabel("L2 Loss [a.u.]")
        
        for idx, trainer in enumerate(trainers):
            if not getattr(trainer.history, "iters", None):
                continue
            color = colors[idx % len(colors)]
            iters = trainer.history.iters
            losses = trainer.history.iter_losses
            if max_iters is not None and max_iters > 0:
                iters = iters[-max_iters:]
                losses = losses[-max_iters:]
            plt.plot(iters, losses, 
                     linestyle="-", 
                     color=color, 
                     label=f"Train Loss {idx+1}")
        
        plt.grid(True)
        if logy:
            plt.yscale("log")
        if logx:
            plt.xscale("log")
        # if xlim is not None:
        #     plt.xlim(None, xlim)
        plt.legend()

        if save_path_iter is not None:
            dirpath = os.path.dirname(save_path_iter) or "."
            os.makedirs(dirpath, exist_ok=True)
            plt.savefig(save_path_iter)
            print(f"Saved iteration plot to {save_path_iter}")
        plt.show()
