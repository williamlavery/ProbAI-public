#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end script: data prep, model construction, trainer orchestration, and optional fit.

Version: 1.2.0
Date: 2025-08-26
Author: W. Lavery
Python: >=3.9

Overview
--------
• Prepares MNIST data (normalized to [-1,1]) and wraps with ProbAIMnistDataset.
• Builds/loads a persistent train/val split.
• Constructs CachedLoaders with deterministic ordering.
• Instantiates TinyVectorField_U (UNet-style) and a FlowTrainer.
• Loads existing trainer if present (idempotent), else creates and saves one.
• Optionally loads a "best" checkpoint and trains further.

Notes
-----
- Commented-out code blocks from your original snippet are preserved.
- This script is safe to re-run: it won’t rebuild the trainer if it already exists.
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

# Models
# ======
from Modules.Models.ModelWrapper import *
from Modules.Models.Unet import *

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

print(f"Train/Val/Test loaders saved to {save_dir}")


# ========================================================================================
# ================================ MODEL CONSTRUCTOR =====================================
# ========================================================================================

# ---------------------------- Model + training parameters ----------------------------
model_params = {
    "modelType":     modelType,
    "Tfeats":        Tfeats,         # Number of features in the vector field
    "ES":            ES,             # Early stopping patience for training
    "lr":            lr,             # Learning rate for the optimizer
    "saveThreshold": saveThreshold,  # Threshold for saving the model
    "device":        device,         # Device to use for training
    # "stamp": date.today().strftime("%B%d"),  # Date stamp for model saving
}

# Instantiate model (current option: UNet)
if modelType == "UNet":
    model = TinyVectorField_U(Tfeats=Tfeats).to(device)
    print("Using UNet model")
else:
    raise ValueError(f"Unknown modelType: {modelType!r}")


# Merge data + model params for pathing
params = {**data_params, **model_params}
path_intro = path_constructor(params)


# ---------------------------- Flow matching wrapper ----------------------------
# Train (adjust epochs as you like)
# train_flow(model, loader, optimizer, epochs=5, log_every=100, device=device)

# Create optimizer
opt = torch.optim.Adam(model.parameters(), lr=lr)

# Paths for trainer + checkpoints
save_best_path = os.path.join(path_intro, f"modelNum_{model_num}")
os.makedirs(save_best_path, exist_ok=True)
trainer_path = os.path.join(save_best_path, "trainer.pt")

# ----------------------------------------------------------------------------- #
# Build or load trainer (DO NOT rebuild if it already exists on disk)          #
# ----------------------------------------------------------------------------- #
if os.path.exists(trainer_path):
    try:
        trainer: FlowTrainer = torch.load(trainer_path, map_location=device)
        print(f"[Trainer] Loaded existing trainer from: {trainer_path}")
        # Ensure trainer uses current code references
        trainer.model = model
        trainer.optimizer = opt
        trainer.device = device
        trainer.model.to(trainer.device)
    except Exception as e:
        print(f"[Trainer] Failed to load existing trainer ({e}). Recreating fresh trainer.")
        trainer = FlowTrainer(
            model, opt,
            device=device,
            log_every=10,
            early_stopping_patience=ES,
            save_threshold=saveThreshold,
            shuffle_pairings=shufflePairings,
        )
        torch.save(trainer, trainer_path)
        print(f"[Trainer] Saved new trainer to: {trainer_path}")
else:
    trainer = FlowTrainer(
        model, opt,
        device=device,
        log_every=10,
        early_stopping_patience=ES,
        save_threshold=saveThreshold,
        shuffle_pairings=shufflePairings,
    )
    torch.save(trainer, trainer_path)
    print(f"[Trainer] Created and saved new trainer to: {trainer_path}")

# ----------------------------------------------------------------------------- #
# Optional: try to load a pre-trained "best" checkpoint                         #
# ----------------------------------------------------------------------------- #
loaded = False
load_best_path = None
save_best_path_check = save_best_path

if load_best:
    # Try alternative ES and epoch choices, newest first
    for ES_c in sorted(ES_check, reverse=True):
        save_best_path_check = save_best_path.replace(f"ES_{ES}", f"ES_{ES_c}")
        # Try explicit epoch list first; if empty, scan directory for available checkpoints
        candidate_epochs = list(sorted(epoch_check, reverse=True)) if epoch_check else []

        # If no explicit epochs provided, discover saved checkpoints
        if not candidate_epochs:
            pattern = os.path.join(save_best_path_check, "model_trainedEpochs*.pt")
            found_epochs = []
            for p in glob.glob(pattern):
                base = os.path.basename(p)
                try:
                    E = int(base.replace("model_trainedEpochs", "").replace(".pt", ""))
                    found_epochs.append(E)
                except Exception:
                    pass
            candidate_epochs = sorted(found_epochs, reverse=True)

        for epoch_c in candidate_epochs:
            load_best_path = os.path.join(save_best_path_check, f"model_trainedEpochs{epoch_c}.pt")
            try:
                trainer.load_best(load_best_path, map_location=device)
                print(f"[Checkpoint] Loaded best checkpoint: {load_best_path}")
                loaded = True
                break  # inner loop
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[Checkpoint] Failed to load {load_best_path}: {e}")

        if loaded:
            break

print(f"============ Loaded pre-trained model Case {CASE} = {loaded} ============")
print("load_best parameters:")
if loaded and load_best_path is not None:
    parse_params_from_path(load_best_path)
else:
    print("  (No checkpoint loaded)")
print("----------------------------------------")
#print("save_best_path:", save_best_path)
parse_params_from_path(save_best_path)
print("==================================================")

# ========================================================================================
# ==================================== MODEL FIT =========================================
# ========================================================================================

if fit_bool:
    if valBool:
        history = trainer.fit(
            loader=train_loader,
            epochs=numEpochs,
            val_loader=val_loader,
            save_best_to=save_best_path,
        )
    else:
        history = trainer.fit(
            loader=train_loader,
            epochs=numEpochs,
            val_loader=None,
            save_best_to=save_best_path,
        )

    # Persist the trainer state after training
    try:
        torch.save(trainer, trainer_path)
        print(f"[Trainer] Updated trainer saved to: {trainer_path}")
    except Exception as e:
        print(f"[Trainer] Warning: failed to save updated trainer: {e}")
