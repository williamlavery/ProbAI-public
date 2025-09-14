#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment configuration parameters.

Version: 1.0.0
Date: 2025-08-26
Author: W Lavery
Python: >=3.9

------------------------------------------------------------------------------
This file defines key hyperparameters and toggles for data preparation
and model training. Import this file into training scripts to ensure
consistent, centralized configuration.
------------------------------------------------------------------------------
"""

CASE: int = 3 # Select CASE 1, 2, or 3 below

# ---------------------- Data parameters ----------------------
# CASE 1
if CASE == 1:
    shufflePairings = False   # Whether to shuffle pairings in the dataset
    cropBool  = False          # Whether to use the crop source distribution

# CASE 2
if CASE == 2:
    shufflePairings = True    # Whether to shuffle pairings in the dataset
    cropBool = True           # Whether to use the crop source distribution

# CASE 3
if CASE == 3:
    shufflePairings = False   # Whether to shuffle pairings in the dataset
    cropBool = True           # Whether to use the crop source distribution


# ---------------------- Model parameters ----------------------
ES: int = 10                   # Early stopping patience (in epochs)
model_num: int = 1             # Model selection index (for experimentation)
numEpochs: int = 50            # Total number of training epochs

# ---------------------- train parameters ----------------------
fit_bool: bool = True           # Whether to fit the model
overwrite: bool = False          # Whether to overwrite existing model checkpoints
load_best: bool = True          # Whether to load the best model checkpoint
epoch_check_max: int = 40        # Maximum epoch to check for best model
ES_check: list[int] = [10]       # Early stopping check interval (in epochs)



# Determined
epoch_check = list(range(epoch_check_max))    # List of epochs to check for best model
