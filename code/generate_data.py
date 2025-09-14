#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end script: data preparation for ProbAI MNIST experiments.

Version: 1.2.0
Date: 2025-08-26
Author: W. Lavery
Python: >=3.9

Overview
--------
• Prepares MNIST data (normalized to [-1,1]) and wraps with ProbAIMnistDataset.
• Builds/loads a persistent train/val split.
• Constructs CachedLoaders with deterministic ordering.

Notes
-----
- Commented-out code blocks from your original snippet are preserved.
- This script is safe to re-run: it won’t rebuild the trainer if it already exists.

TO DO
-----
- Currently overkill on reports and simulate.py could make use of this script.
"""

from __future__ import annotations

# Allow imports from project root if needed
from sys import path as _syspath
_syspath.append("../")

import os
import glob
from datetime import date

import torch
from torchvision import datasets, transforms

# Utils
# =====
from Modules.Utils.dt8122_snippets import *        # ProbAIMnistDataset, build_or_load_split, make_order, path_constructor, FlowTrainer, etc.
# from Modules.Utils.helpers import *
from Modules.Utils.dataClass_helpers import *
from Modules.Utils.DataLoader import *
from Modules.Utils.dataPlot_helpers import *
from Modules.Utils.miscellaneous_helpers import *


# Parameters
# ==========
from config.config_static import *
from config.config_dynamic import *


# ---------------------- Device selection ----------------------
device = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
# device = torch.device("cpu")  # Force CPU for compatibility
print("Using device:", device)



# =================================================================
# =========================== DATA PREP ===========================
# =================================================================


configs = {
    "c1": {"shufflePairings": False,
           "cropBool":False
           },
    "c2": {"shufflePairings": True,
           "cropBool":True
           },
    "c3": {"shufflePairings": False,
           "cropBool":True
           }
}

for config_name, config in configs.items():
    shufflePairings = config["shufflePairings"]
    cropBool = config["cropBool"]
    # ---------------------- Data parameters ----------------------
    data_params = {
        "shufflePairings": shufflePairings,
        "cropBool":        cropBool,
        "valFrac":         valFrac,
        "splitSeed":        splitSeed,
        "initialShuffleSeed": initialShuffleSeed,
        "batchSize":       batchSize,
        "valBool":         valBool,
    }

    # ---------------------- 1) Transforms ----------------------
    # Normalize MNIST digits from [0,1] → [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # ---------------------- 2) Download MNIST ----------------------
    mnist_train = datasets.MNIST(root="./data/train", train=True,  download=True, transform=transform)
    mnist_test  = datasets.MNIST(root="./data/test",  train=False,  download=True, transform=transform)

    # ---------------------- 3) Wrap with custom dataset ----------------------
    probai_dataset_train = ProbAIMnistDataset(mnist_train, crop_noise=cropBool)
    probai_dataset_test  = ProbAIMnistDataset(mnist_test,  crop_noise=cropBool)

    # ---------------------- 4) Build or load persistent split ----------------------
    train_ds, val_ds, train_idx, val_idx = build_or_load_split(
        probai_dataset_train, val_frac=valFrac, seed=splitSeed
    )

    # ---------------------- Store data in tensors ----------------------
    x0_store_train, x1_store_train = map(list, zip(*train_ds))
    x0_store_val,   x1_store_val   = map(list, zip(*val_ds))
    x0_store_test                  = [x0 for (x0, _) in probai_dataset_test]
    x1_store_test                  = [x1 for (_,  x1) in probai_dataset_test]


    # ---------------------- 5) DataLoaders ----------------------
    N = len(x0_store_train)
    order     = make_order(N, seed=initialShuffleSeed,               epoch=0)
    order_val = make_order(len(x0_store_val), seed=initialShuffleSeed, epoch=0)
    order_test = list(range(len(x0_store_test)))  # no shuffling for test

    train_loader = CachedLoader(
        x0_store_train, x1_store_train,
        batch_size=batchSize,
        order=order,
        drop_last=True,
        device=device,        # optional: move batches to device
        non_blocking=False,
        shuffle_pairs=shufflePairings
    )

    val_loader = CachedLoader(
        x0_store_val, x1_store_val,
        batch_size=batchSize,
        order=order_val,
        drop_last=True,
        device=device,
        non_blocking=False,
        shuffle_pairs=shufflePairings
    )


    test_loader = CachedLoader(
        x0_store_test, x1_store_test,
        batch_size=batchSize,
        order=order_test,
        drop_last=False,   # keep all test samples
        device=device,
        non_blocking=False,
    )

    # ---------------------- 6) Save loaders (optional, kept commented) ----------------------
    save_dir = path_constructor(data_params, "data_processed")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_loader, os.path.join(save_dir, "train_loader.pt"))
    torch.save(val_loader,   os.path.join(save_dir, "val_loader.pt"))
    torch.save(test_loader,  os.path.join(save_dir, "test_loader.pt"))

    print("\n" + "="*50)
    print(f"Config: {config_name}")
    print(f"Train/Val/Test loaders saved to {save_dir}")
    print("\n" + "="*50)