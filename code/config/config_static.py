#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment configuration parameters.

Version: 1.0.0
Date: 2025-08-26
Author: Your Name <you@example.com>
Python: >=3.9

------------------------------------------------------------------------------
This file defines key hyperparameters and toggles for dataset preparation
and model training. Import this file into training scripts to ensure
consistent, centralized configuration.
------------------------------------------------------------------------------
"""

# ---------------------- Data parameters ----------------------
valFrac: float = 0.10              # Fraction of data to use for validation
splitSeed: int = 42                # Random seed for reproducibility of splits
initialShuffleSeed: int = 12345    # Random seed for initial shuffling of (x0, x1) pairs
batchSize: int = 32                # Batch size for training
valBool: int = 1                   # Whether to use a validation set (1=True, 0=False)

# ---------------------- Model parameters ----------------------
Tfeats: int = 11                   # Number of time-feature channels (see `Unet.py`)
modelType: str = "UNet"            # Currently only "UNet" implemented
lr: float = 1e-3                   # Adam learning rate
saveThreshold: float = 0.99        # Save model if loss reduces by this factor
