#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowTrainer: Minimal, pragmatic trainer for pairwise *flow matching* with PyTorch.

Version: 1.1.0
Date: 2025-08-26
Author: W. Lavery 
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• A light-weight training loop (`FlowTrainer`) for models that learn *velocity
  fields* using Flow Matching (FM) on paired data (x0, x1).
• History tracking (iterations, epoch averages, optional validation).
• "Best" checkpoint saving/loading with early stopping (epoch-based).
• Drop-in hooks for *noisy* path training (currently commented out and kept).

------------------------------------------------------------------------------
The theory (short, practitioner-friendly)
------------------------------------------------------------------------------
Flow Matching (FM) trains a time-conditional vector field v_θ(x, t) to match a
*target* vector field v*(x, t) along a prescribed reference path x(t) that
connects a source distribution to a data distribution. In the simplest *linear
interpolant* setting used here:

    x(t)  = (1 − t) * x0 + t * x1                 with  t ~ Uniform[0, 1]
    v*(t) = d/dt x(t) = (x1 − x0)

Given a pair (x0, x1), we sample t, form x_t = x(t), and train v_θ to minimize

    L(θ) = E_{(x0,x1), t} [ || v_θ(x_t, t) − (x1 − x0) ||^2 ].

This objective teaches the model to predict the *instantaneous velocity* that
moves x_t forward along the path from x0 to x1.

Why add noise? (commented blocks below)
---------------------------------------
Deterministic linear paths can be enriched by injecting Gaussian noise around
the path mean μ_t (often the same as x_t for the linear path) with a schedule
σ(t). For Gaussian perturbations x̃_t = μ_t + σ(t) ε, the *conditional*
target velocity includes an extra correction term:

    v*(x̃_t, t) = (x1 − x0) + (d/dt log σ(t)) * (x̃_t − μ_t)

This yields a denoising-like FM objective that improves robustness and
expressivity. The code contains ready-made (commented) stubs for σ schedules
and the corresponding target.

"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # (kept; useful for external plotting)
from pathlib import Path


@dataclass
class TrainHistory:
    """
    Container recording training/validation traces.

    Attributes
    ----------
    iters : List[int]
        Global iteration indices (1-based, monotonically increasing).
    iter_losses : List[float]
        Per-iteration training losses.
    epochs : List[int]
        Epoch counters (0-based internal epoch number).
    epoch_losses : List[float]
        Per-epoch average training loss.
    val_epoch_losses : List[Optional[float]]
        Per-epoch average validation loss; None when no val loader is used.
    logs : List[Dict[str, Any]]
        Mixed granular logs: entries of type "iter", "epoch", or "early_stop".
    """
    iters: List[int] = field(default_factory=list)
    iter_losses: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    epoch_losses: List[float] = field(default_factory=list)
    val_epoch_losses: List[Optional[float]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)

# For adding noise to linear interpolation - currently not implemented
# def sigma_bump(t, sigma_min=0.02, sigma_bump=0.4):
#      """
#      "Bump"-shaped schedule:
#          sigma(t) = sigma_min + sigma_bump * 4 t (1 - t)
#      d/dt log sigma(t) is returned for the conditional target term.
#      Parameters
#      ----------
#      t : torch.Tensor
#          Shape [B, 1, 1, 1], values in [0, 1].
#      sigma_min : float
#      sigma_bump : float
#      Returns
#      -------
#      sigma : torch.Tensor
#      dlog_sigma_dt : torch.Tensor
#      """
#      # t: [B,1,1,1] in [0,1]
#      sigma = sigma_min + sigma_bump * (4.0 * t * (1.0 - t))
#      ds_dt = sigma_bump * (4.0 * (1.0 - 2.0 * t))
#      dlog_sigma_dt = ds_dt / sigma
#      return sigma, dlog_sigma_dt

# def sigma_linear(t, sigma0=0.4, sigma1=0.05):
#      """
#      Linear schedule from sigma0 to sigma1 over t in [0, 1].
#      Returns sigma(t) and d/dt log sigma(t).
#      """
#      print("Using linear")
#      sigma = (1.0 - t) * sigma0 + t * sigma1
#      ds_dt = (sigma1 - sigma0) * torch.ones_like(t)
#      dlog_sigma_dt = ds_dt / sigma
#      return sigma, dlog_sigma_dt


class FlowTrainer:
    """
    Train a time-conditional velocity model with Flow Matching on paired data.

    The trainer expects a model with signature:
        model(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor
    returning a tensor with the same shape as x_t that estimates the target
    velocity field v*(x_t, t).

    Parameters
    ----------
    model : torch.nn.Module
        Time-conditional velocity model v_θ(x, t).
    optimizer : torch.optim.Optimizer
        Optimizer configured on `model.parameters()`.
    device : str, default "cpu"
        Device string understood by torch (e.g., "cuda", "cuda:0", "mps", "cpu").
    log_every : int, default 100
        Print a progress line every `log_every` iterations within an epoch.
    line_width : int, default 120
        Width used to right-pad progress log lines.
    save_threshold : float, default 0.95
        Multiplicative improvement threshold; save best if `metric < best * threshold`.
        (Smaller-is-better metrics.)
    early_stopping_patience : int, default 200
        Number of *epochs* without improvement before early stopping triggers.
    shuffle_pairings : bool, default True
        If loaders expose `shuffle_pairings(seed, epoch)`, use it prior to training.

    Notes
    -----
    • Validation (if provided) is executed *once per epoch* at epoch end.
    • Early stopping is evaluated on the monitored epoch metric
      (validation if available, otherwise training average).
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = "cpu",
                 log_every: int = 100,
                 line_width: int = 120,
                 save_threshold: float = 0.95,
                 early_stopping_patience: Optional[int] = 200,
                 shuffle_pairings: bool = True):  # interpreted as *epochs* here (since we validate at epoch end)
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.log_every = log_every
        self.line_width = line_width
        self.history = TrainHistory()
        self.save_threshold = save_threshold  # multiplicative (smaller-is-better)
        self.shuffle_pairings = shuffle_pairings  # whether to shuffle pairings in the dataset

        # best checkpoint tracking
        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_iter = None
        self.best_is_val = True               # we monitor validation at epoch end when provided
        self.best_path = None

        # epoch counting persists
        self.epoch = -1

        # early stopping state (epoch-based now)
        self.early_stopping_patience = int(early_stopping_patience) if early_stopping_patience is not None else 0
        self.early_stop_wait = 0              # counts epochs w/o improvement
        self.early_stop_triggered = False
        self.early_stop_reason = None

        self.model.to(self.device)

    # ---------- utilities ----------
    def _format_eta(self, seconds: Optional[float]) -> str:
        """Format a seconds float into M:SS or H:MM:SS; returns '--:--' if unknown."""
        if seconds is None or seconds != seconds or seconds < 0:
            return "--:--"
        m, s = divmod(int(seconds + 0.5), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def _log_line(self,
                  epoch: int,
                  epochs: int,
                  it: int,
                  total_in_epoch: Optional[int],
                  loss_val: float,
                  eta_epoch_sec: Optional[float],
                  eta_total_sec: Optional[float]) -> None:
        """Print a single, right-padded progress line."""
        total_str = f"/{total_in_epoch}" if total_in_epoch is not None else "/??"
        msg = (
            f"Epoch {epoch}/{epochs} | iter {it:04d}{total_str} | "
            f"train {loss_val:.6f}"
        )
        msg += (
            f" | ETA(ep) {self._format_eta(eta_epoch_sec)}"
            f" | ETA(total) {self._format_eta(eta_total_sec)}"
            f" | Trained epochs {self.epoch}"
        )
        print(msg.ljust(self.line_width), end="\r", flush=True)

    def _update_best_and_es_epoch(self,
                                  metric: float,
                                  epoch: int,
                                  it: int,
                                  is_val: bool,
                                  save_best_to: Optional[str]) -> None:
        """
        Update best_* using per-epoch metric and handle epoch-based early stopping.

        Parameters
        ----------
        metric : float
            The epoch metric to monitor (smaller-is-better).
        epoch : int
            1-based epoch index for user display.
        it : int
            Global iteration at time of update (for bookkeeping only).
        is_val : bool
            True if `metric` came from validation, False if from training.
        save_best_to : Optional[str]
            Directory to save best checkpoints. If None, no saving occurs.
        """
        best = self.best_loss
        improved = metric < best

        if improved:
            self.best_loss = float(metric)
            self.best_epoch = int(epoch)
            self.best_iter  = int(it)
            self.best_is_val = bool(is_val)

            if metric < best * self.save_threshold and save_best_to is not None:
                self.save_best(save_best_to)
                self.early_stop_wait = 0
                self.early_stop_triggered = False
                self.early_stop_reason = None
            else:
                self.early_stop_wait += 1
        else:
            if self.early_stopping_patience > 0:
                self.early_stop_wait += 1
                if self.early_stop_wait >= self.early_stopping_patience:
                    self.early_stop_triggered = True
                    self.early_stop_reason = (
                        f"No improvement in monitored loss for "
                        f"{self.early_stopping_patience} epoch(s)."
                    )

    # ---------- core APIs ----------
    def fit(self,
            loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
            epochs: int = 5,
            val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
            save_best_to: Optional[str] = None) -> TrainHistory:
        """
        Train for the given number of epochs using the Flow Matching objective.

        Mechanics
        ---------
        • At each iteration, sample t ~ U[0,1], form x_t = (1−t) x0 + t x1,
          and set target velocity v_target = (x1 − x0).
        • Compute MSE: || v_θ(x_t, t) − v_target ||² and update θ.
        • (Optional) Run validation once at the end of each epoch.
        • (Optional) Save "best" checkpoint if monitored metric improved
          sufficiently (multiplicative threshold).
        • Early stopping patience is measured in *epochs*.

        Parameters
        ----------
        loader : Iterable[(x0, x1)]
            Training iterator yielding pairs of tensors with identical shapes.
            If it (or its underlying dataset) provides a .shuffle(...) or
            .shuffle_pairings(...), they will be called when available.
        epochs : int, default 5
            Number of epochs to train.
        val_loader : Optional[Iterable[(x0, x1)]]
            Validation iterator; averaged once per epoch if provided.
        save_best_to : Optional[str]
            Directory to save best checkpoints; created if missing.

        Returns
        -------
        TrainHistory
            Filled history with per-iter and per-epoch traces.
        """
        self.model.train()
        self.global_it = getattr(self, "global_it", 0)

        steps_per_epoch = len(loader) if hasattr(loader, "__len__") else None
        total_steps_planned = epochs * steps_per_epoch if steps_per_epoch is not None else None
        global_iter_time_sum = 0.0
        global_iter_count = 0

        # Optional pairing shuffles before training begins
        # if self.shuffle_pairings and self.epoch == -1:
        #     if hasattr(loader, "shuffle_pairings"):
        #         print("Shuffling pairings in the dataset (pre-training)")
        #         loader.shuffle_pairings(seed=0, epoch=self.epoch)
        #     if val_loader is not None and hasattr(val_loader, "shuffle_pairings"):
        #         val_loader.shuffle_pairings(seed=0, epoch=self.epoch)

        # Catch before training (if model has been pretrained)
        if self.early_stop_wait >= self.early_stopping_patience:
            self.early_stop_triggered = True
            self.early_stop_reason = (
                    f"No improvement in monitored loss for "
                    f"{self.early_stopping_patience} epoch(s)."
                )

        for epch in range(epochs):

            self.epoch += 1
            epoch_index = self.epoch + 1  # human-friendly

            # Gentle feature-detection for loader shuffles
            if hasattr(loader, "shuffle"):
                loader.shuffle(seed=0, epoch=self.epoch)
            if val_loader is not None and hasattr(val_loader, "shuffle"):
                val_loader.shuffle(seed=0, epoch=self.epoch)

            epoch_start_time = time.time()
            running_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_iter_count = 0

            iter_time_sum = 0.0
            last_iter_t = time.time()
            total_in_epoch = steps_per_epoch

            for it, (x0, x1) in enumerate(loader, 1):
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)

                B = x0.shape[0]
                torch.manual_seed(self.global_it + it)  # ensure reproducibility
                t = torch.rand(B, 1, 1, 1, device=self.device)

                # Deterministic linear path and target
                x_t = (1.0 - t) * x0 + t * x1
                v_target = (x1 - x0)

                # ============ For adding noise to trajectory  ============
                # Path mean
                # mu_t = (1.0 - t) * x0 + t * x1
                #
                # # Choose a schedule
                # sigma_t, dlog_sigma_dt = sigma_bump(t)     # or: sigma_linear(t)
                #
                # # Sample noisy state along the path
                # eps = torch.randn_like(x0)
                # x_t = mu_t + sigma_t * eps
                #
                # # Conditional velocity target for Gaussian path
                # v_target = (x1 - x0) + dlog_sigma_dt * (x_t - mu_t)
                # ========================================================

                v_pred = self.model(x_t, t)
                loss = F.mse_loss(v_pred, v_target)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # bookkeeping (train)
                loss_val = float(loss.item())
                self.global_it += 1
                running_loss += loss_val
                epoch_loss_sum += loss_val
                epoch_iter_count += 1

                self.history.iters.append(self.global_it)
                self.history.iter_losses.append(loss_val)
                self.history.logs.append({
                    "type": "iter",
                    "epoch": self.epoch,
                    "iter_in_epoch": it,
                    "global_iter": self.global_it,
                    "loss": loss_val,
                    "time": time.time(),
                })

                # ETA updates
                now_t = time.time()
                dt = (now_t - last_iter_t)
                last_iter_t = now_t

                iter_time_sum += dt
                avg_iter_time_ep = iter_time_sum / max(1, epoch_iter_count)
                eta_epoch = ((total_in_epoch - it) * avg_iter_time_ep) if total_in_epoch is not None else None

                global_iter_time_sum += dt
                global_iter_count += 1
                avg_iter_time_global = global_iter_time_sum / max(1, global_iter_count)
                steps_done = ((epoch_index - 1) * steps_per_epoch + it) if steps_per_epoch is not None else None
                eta_total = ((total_steps_planned - steps_done) * avg_iter_time_global) if (total_steps_planned is not None and steps_done is not None) else None

                if self.log_every and (it % self.log_every == 0):
                    self._log_line(epch, epochs, it, total_in_epoch, running_loss / self.log_every, eta_epoch, eta_total)
                    running_loss = 0.0

            # ---------- epoch end: compute train avg ----------
            epoch_avg = epoch_loss_sum / max(1, epoch_iter_count)
            epoch_duration = time.time() - epoch_start_time

            self.history.epochs.append(self.epoch)
            self.history.epoch_losses.append(epoch_avg)

            # ---------- epoch end: run validation once ----------
            if val_loader is not None:
                val_epoch_avg = self.validate(val_loader)
                self.history.val_epoch_losses.append(val_epoch_avg)

                epoch_msg = (
                    f"Epoch {epoch_index} done | "
                    f"avg train {epoch_avg:.6f} | "
                    f"avg val {val_epoch_avg:.6f} | "
                    f"duration {self._format_eta(epoch_duration)} | "
                    f"Total # trained epochs {self.epoch} | "
                    f"Trigger {self.early_stop_wait }"
                )

                # update best + early stopping on validation loss
                self._update_best_and_es_epoch(
                    metric=val_epoch_avg,
                    epoch=epoch_index,
                    it=self.global_it,
                    is_val=True,
                    save_best_to=save_best_to
                )
            else:
                # no validation set: track by training epoch avg
                self.history.val_epoch_losses.append(None)
                epoch_msg = (
                    f"Epoch {epoch_index} done | "
                    f"avg train {epoch_avg:.6f} | "
                    f"duration {self._format_eta(epoch_duration)} | "
                    f"Total # trained epochs {self.epoch} | "
                    f"Trigger {self.early_stop_wait }"
                )

                self._update_best_and_es_epoch(
                    metric=epoch_avg,
                    epoch=epoch_index,
                    it=self.global_it,
                    is_val=False,
                    save_best_to=save_best_to
                )

            print(epoch_msg.ljust(self.line_width))

            self.history.logs.append({
                "type": "epoch",
                "epoch": self.epoch,
                "avg_loss": epoch_avg,
                "val_avg_loss": self.history.val_epoch_losses[-1],
                "duration_sec": epoch_duration,
                "time": time.time(),
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "best_iter": self.best_iter,
                "best_is_val": self.best_is_val,
            })

            # early stopping check at epoch boundary
            if self.early_stop_triggered:
                stop_msg = f"Early stopping at epoch {epoch_index}: {self.early_stop_reason}"
                print(stop_msg.ljust(self.line_width))
                self.history.logs.append({
                    "type": "early_stop",
                    "epoch": self.epoch,
                    "iter_in_epoch": None,
                    "global_iter": self.global_it,
                    "reason": self.early_stop_reason,
                    "best_loss": self.best_loss,
                    "best_epoch": self.best_epoch,
                    "best_iter": self.best_iter,
                    "time": time.time(),
                })
                break

        return self.history

    @torch.no_grad()
    def validate(self,
                 loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        Average MSE over the whole validation loader (model state preserved).
        Called *at the end of each epoch* when `val_loader` is provided.

        Parameters
        ----------
        loader : Iterable[(x0, x1)]
            Validation iterator yielding pairs of tensors with identical shapes.

        Returns
        -------
        float
            Mean validation MSE across the iterator.
        """
        was_training = self.model.training
        self.model.eval()

        total = 0.0
        n = 0
        for x0, x1 in loader:
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)

            B = x0.shape[0]
            t = torch.rand(B, 1, 1, 1, device=self.device)

            x_t = (1.0 - t) * x0 + t * x1
            v_target = (x1 - x0)
            v_pred = self.model(x_t, t)

            loss = F.mse_loss(v_pred, v_target, reduction="mean")
            total += float(loss.item())
            n += 1

        if was_training:
            self.model.train()
        return total / max(1, n)

    # ---------- checkpoint helpers ----------

    def save_best(self, path: str) -> None:
        """
        Save the current *best* model/optimizer and brief metadata to `path`.
        Ensures the parent directory exists.

        The file name includes the current internal epoch counter:
            {path}/model_trainedEpochs{self.epoch}.pt
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"model_trainedEpochs{self.epoch}.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(getattr(self, "epoch", 0)),  # convenience for resuming
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "best_iter": self.best_iter,
            "best_is_val": self.best_is_val,
            "history": {
                "iters": getattr(self.history, "iters", []),
                "iter_losses": getattr(self.history, "iter_losses", []),
                "epochs": getattr(self.history, "epochs", []),
                "epoch_losses": getattr(self.history, "epoch_losses", []),
                "val_epoch_losses": getattr(self.history, "val_epoch_losses", []),
            },
            "early_stopping": {
                "patience": self.early_stopping_patience,  # epochs
                "wait": self.early_stop_wait,
                "triggered": self.early_stop_triggered,
                "reason": self.early_stop_reason,
            },
        }, path)

        self.best_path = str(path)

    def load_best(self,
                  path: Optional[str] = None,
                  map_location: Optional[str] = None,
                  strict: bool = True,
                  load_history: bool = True) -> Dict[str, Any]:
        """
        Load best checkpoint, including optimizer and (optionally) training history.

        Parameters
        ----------
        path : Optional[str]
            Path to a specific checkpoint file. If None, uses `self.best_path`.
        map_location : Optional[str]
            Device mapping for `torch.load` (e.g., "cpu").
        strict : bool, default True
            Whether to strictly enforce that the keys in `state_dict` match.
        load_history : bool, default True
            Whether to restore training history into `self.history`.

        Returns
        -------
        Dict[str, Any]
            The raw checkpoint dictionary.
        """
        ckpt_path = path or self.best_path
        if ckpt_path is None:
            raise ValueError("No checkpoint path provided and no previous best save path available.")

        ckpt = torch.load(ckpt_path, map_location=map_location)

        # --- weights & optimizer ---
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in ckpt and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                # Optimizer shapes/param groups changed; keep going with fresh optimizer
                pass

        # --- best metrics / bookkeeping ---
        self.best_loss = float(ckpt.get("best_loss", getattr(self, "best_loss", float("inf"))))
        self.best_epoch = ckpt.get("best_epoch", getattr(self, "best_epoch", None))
        self.best_iter  = ckpt.get("best_iter", getattr(self, "best_iter", None))
        self.best_is_val = bool(ckpt.get("best_is_val", getattr(self, "best_is_val", False)))
        self.best_path = ckpt_path

        es = ckpt.get("early_stopping", {})
        if isinstance(es, dict):
            self.early_stopping_patience = int(es.get("patience", getattr(self, "early_stopping_patience", 0)))
            self.early_stop_wait = int(es.get("wait", getattr(self, "early_stop_wait", 0)))
            # self.early_stop_triggered = bool(es.get("triggered", getattr(self, "early_stop_triggered", False)))
            # self.early_stop_reason = es.get("reason", getattr(self, "early_stop_reason", None))

        # --- history (NEW) ---
        if load_history:
            hist = ckpt.get("history", None)

            # Ensure self.history exists (very lightweight fallback)
            if getattr(self, "history", None) is None:
                class _History:  # minimal shim if your project doesn't have a History class here
                    def __init__(self):
                        self.iters = []
                        self.iter_losses = []
                        self.epochs = []
                        self.epoch_losses = []
                        self.val_epoch_losses = []
                self.history = _History()

            if isinstance(hist, dict):
                # Replace in-memory history with the checkpointed one
                self.history.iters = list(hist.get("iters", []))
                self.history.iter_losses = list(hist.get("iter_losses", []))
                self.history.epochs = list(hist.get("epochs", []))
                self.history.epoch_losses = list(hist.get("epoch_losses", []))
                self.history.val_epoch_losses = list(hist.get("val_epoch_losses", []))

                # Try to restore current epoch counter for a clean resume
                ckpt_epoch = ckpt.get("epoch", None)
                if ckpt_epoch is not None:
                    self.epoch = int(ckpt_epoch)
                elif getattr(self.history, "epochs", None):
                    # Fall back to last logged epoch if present
                    try:
                        self.epoch = int(self.history.epochs[-1])
                    except Exception:
                        pass

        return ckpt

    # optional helpers
    def get_history(self) -> TrainHistory:
        """Return the internal TrainHistory object (live reference)."""
        return self.history

    def state_dict(self) -> Dict[str, Any]:
        """
        Lightweight serialization of trainer state (not model weights).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing history, device, logging config, best metrics,
            early stopping state, and the current epoch counter.
        """
        return {
            "history": {
                "iters": self.history.iters,
                "iter_losses": self.history.iter_losses,
                "epochs": self.history.epochs,
                "epoch_losses": self.history.epoch_losses,
                "val_epoch_losses": self.history.val_epoch_losses,
            },
            "log_every": self.log_every,
            "device": self.device,
            "best": {
                "loss": self.best_loss,
                "epoch": self.best_epoch,
                "iter": self.best_iter,
                "is_val": self.best_is_val,
                "path": self.best_path,
            },
            "early_stopping": {
                "patience": self.early_stopping_patience,  # epochs
                "wait": self.early_stop_wait,
                "triggered": self.early_stop_triggered,
                "reason": self.early_stop_reason,
            },
            "trained_epochs": self.epoch,
        }
