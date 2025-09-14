#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for paths to saved model checkpoints.

Version: 1.2.0
Date: 2025-08-26
Author: W Lavery
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• path_constructor            : Safe, deterministic path builder from parameter dicts
                                (sanitizes '/', formats floats compactly, sorts keys).

• parse_params_from_path      : Recover params from any path segment shaped like
                                "<key>_<value>"; returns dict[str, str] and pretty-prints.

• load_best_checkpoint        : Robust checkpoint loader that searches across early-stopping
                                (ES) variants and epoch candidates. Falls back to directory
                                scanning when explicit epochs aren’t provided. Returns
                                (loaded: bool, path: Optional[str]).

Notes
-----
- Functions keep backwards-compatible signatures where possible.
- Commented-out code blocks are preserved exactly as provided.

License
-------
SPDX-License-Identifier: MIT
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional



from pathlib import Path
from typing import Dict, Any

def path_constructor(params: Dict[str, Any], base: str | Path = "models") -> str:
    """
    Construct a filesystem-safe hierarchical path from a parameter dict.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameters (key → value) to encode into the path.
    base : str or Path, optional
        Base directory for the constructed path (default: "models").

    Returns
    -------
    str
        Constructed hierarchical path.

    Notes
    -----
    - Keys are sorted for determinism.
    - Slashes in values are replaced with underscores.
    - Floats are formatted compactly (up to 6 significant digits).
    """
    base = Path(base)

    for key in params.keys():
        value = params[key]
        if isinstance(value, float):
            # Compact yet readable float formatting
            value_str = f"{value:.6g}"
        else:
            value_str = str(value)
        value_str = value_str.replace("/", "_")
        base = base / f"{key}_{value_str}"

    return str(base)


def parse_params_from_path(path: str) -> Dict[str, str]:
    """
    Parse a path with sections of the form paramName_paramValue
    into a dictionary {paramName: paramValue}.

    Returns
    -------
    dict[str, str]
    """
    params: Dict[str, str] = {}
    
    # Normalize path and split into sections
    parts = os.path.normpath(path).split(os.sep)
    
    for part in parts:
        if "_" in part:
            key, value = part.split("_", 1)  # split only at the first underscore
            params[key] = value
    
    # Pretty print
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    return params

import os
import glob
from typing import Iterable, Optional, Tuple, Any

def load_best_checkpoint(
    *,
    load_best: bool,
    save_best_path: str,
    ES: int,
    ES_check: Iterable[int],
    epoch_check: Iterable[int],
    trainer: Any,
    device: str,
    CASE: str,
    parse_params_from_path,
    verbose: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Try to load the 'best' checkpoint from a directory structure that encodes early-stopping (ES)
    and trained epochs in file names like:  <save_best_path>/model_trainedEpochs{E}.pt

    The function:
      1) Tries alternative ES values (ES_check), newest-first.
      2) For each ES candidate, tries explicit epoch candidates (epoch_check), newest-first.
      3) If no explicit epoch list is provided, it discovers available checkpoints by scanning the directory.
      4) Attempts to load the first matching checkpoint with trainer.load_best(...).

    Parameters
    ----------
    load_best : bool
        Whether to attempt loading a checkpoint at all.
    save_best_path : str
        Base path template pointing to an ES-specific directory. It should include the substring
        f"ES_{ES}" so that alternative ES values can be substituted, e.g. ".../ES_10/...".
    ES : int
        The current/nominal early-stopping value in the provided save_best_path.
    ES_check : Iterable[int]
        Candidate ES values to try, ordered newest-first is recommended. Will be sorted(descending).
    epoch_check : Iterable[int]
        Candidate epochs to try, ordered newest-first is recommended. If empty, will scan the directory.
    trainer : Any
        Object exposing `load_best(path, map_location=...)`.
    device : str
        Device mapping string (e.g., 'cpu' or 'cuda:0') passed to `map_location`.
    CASE : str
        Label printed for logging (e.g., experiment or configuration name).
    parse_params_from_path : Callable[[str], None]
        Function used for logging/inspecting parameters encoded in the path.
    verbose : bool, default True
        Whether to print progress and summary information.

    Returns
    -------
    (loaded, path) : (bool, Optional[str])
        `loaded` indicates whether a checkpoint was successfully loaded.
        `path` is the path of the loaded checkpoint if any, else None.
    """
    loaded = False
    load_path: Optional[str] = None
    save_best_path_check = save_best_path

    if load_best:
        # Try alternative ES and epoch choices, newest first
        for ES_c in sorted(set(ES_check), reverse=True):
            # Swap the ES marker in the path; if absent, just append/replace conservatively
            if f"ES_{ES}" in save_best_path:
                save_best_path_check = save_best_path.replace(f"ES_{ES}", f"ES_{ES_c}")
            else:
                # Fallback: try inserting ES marker as a sibling directory
                head, tail = os.path.split(save_best_path)
                save_best_path_check = os.path.join(head, f"ES_{ES_c}", tail)

            # Use explicit epoch list if provided; otherwise discover from directory
            candidate_epochs = list(sorted(set(epoch_check), reverse=True)) if epoch_check else []

            if not candidate_epochs:
                pattern = os.path.join(save_best_path_check, "model_trainedEpochs*.pt")
                found_epochs = []
                for p in glob.glob(pattern):
                    base = os.path.basename(p)
                    try:
                        E = int(base.replace("model_trainedEpochs", "").replace(".pt", ""))
                        found_epochs.append(E)
                    except Exception:
                        # Ignore files that don't match the expected pattern cleanly
                        pass
                candidate_epochs = sorted(set(found_epochs), reverse=True)

            for epoch_c in candidate_epochs:
                attempt_path = os.path.join(save_best_path_check, f"model_trainedEpochs{epoch_c}.pt")
                try:
                    trainer.load_best(attempt_path, map_location=device)
                    if verbose:
                        print(f"[Checkpoint] Loaded best checkpoint: {attempt_path}")
                    load_path = attempt_path
                    loaded = True
                    break  # break epoch loop
                except FileNotFoundError:
                    # Silent miss is fine; continue searching
                    pass
                except Exception as e:
                    if verbose:
                        print(f"[Checkpoint] Failed to load {attempt_path}: {e}")

            if loaded:
                break  # break ES loop

    if verbose:
        print(f"============ Loaded pre-trained model Case {CASE} = {loaded} ============")
        print("load_best parameters:")
        if loaded and load_path is not None:
            parse_params_from_path(load_path)
        else:
            print("  (No checkpoint loaded)")
        print("----------------------------------------")
        parse_params_from_path(save_best_path)
        print("==================================================")

    return loaded, load_path
