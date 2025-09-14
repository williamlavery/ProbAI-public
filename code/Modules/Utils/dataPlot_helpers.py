#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for flow sampling/integration and visualization

Version: 1.2.0
Date: 2025-08-26
Author: W. Lavery 
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• sample_flow_x(...), *_all   : Heun (2nd-order) integrators for vector-field models.
• show_trajectory_collated    : Panel visualization of an entire trajectory.
• show_clean_collated         : Grid of clean digits with optional save.
• show_collated               : Generic grid helper for [-1,1]-scaled images.
• save_trajectory_gif         : Export a GIF from a trajectory.
• save_trajectory_strip       : Export a single figure with frames arranged horizontally.

Notes
-----
- Functions keep backwards-compatible signatures where possible.
- Commented-out code blocks are preserved exactly as provided.

"""
import os
import matplotlib.pyplot as plt
import math
import torch
from torchvision.utils import make_grid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import matplotlib.animation as animation
from IPython.display import Image, display



@torch.no_grad()
def sample_flow_x(model, x: torch.Tensor, steps: int = 60, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Integrate a vector field forward in time using Heun's method (2nd order).

    Parameters
    ----------
    model : nn.Module
        Callable with signature v = model(x_t, t) where t is [B,1,1,1].
    x : torch.Tensor
        Initial states [N, C, H, W].
    steps : int, default 60
        Number of integration steps from t=0 to t=1.
    device : str | torch.device
        Device to place states and time tensors on.

    Returns
    -------
    torch.Tensor : Final states in [-1, 1] with shape [N, C, H, W].
    """
    model.eval()
    x = x.to(device)
    n = x.size(0)
    dt = 1.0 / max(1, int(steps))

    for i in range(int(steps)):
        # Use scalar times broadcasted to (n,1,1,1) for each step
        t0 = torch.full((n, 1, 1, 1), i * dt, device=device, dtype=x.dtype)
        v0 = model(x, t0)
        x_euler = x + dt * v0

        t1 = torch.full((n, 1, 1, 1), (i + 1) * dt, device=device, dtype=x.dtype)
        v1 = model(x_euler, t1)

        x = x + dt * 0.5 * (v0 + v1)

    return x.clamp(-1, 1)



@torch.no_grad()
def sample_flow_x_all(model, x: torch.Tensor, steps: int = 60, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Run Heun's method (2nd-order) flow integration and return *all* states x_t.
    
    Args
    ----
    model : nn.Module
        Vector field predictor v(x_t, t).
    x : torch.Tensor
        Initial batch [N, C, H, W].
    steps : int
        Number of integration steps.
    device : str | torch.device
        Device to run on.
    
    Returns
    -------
    torch.Tensor
        Tensor of shape [steps+1, N, C, H, W], containing trajectory:
        x[0] = initial, x[-1] = final.
    """
    model.eval()
    x = x.to(device)
    n = x.size(0)
    dt = 1.0 / max(1, int(steps))

    traj = [x.clamp(-1, 1)]  # store initial state

    for i in range(int(steps)):
        t0 = torch.full((n, 1, 1, 1), i * dt, device=device, dtype=x.dtype)
        v0 = model(x, t0)
        x_euler = x + dt * v0

        t1 = torch.full((n, 1, 1, 1), (i + 1) * dt, device=device, dtype=x.dtype)
        v1 = model(x_euler, t1)

        x = x + dt * 0.5 * (v0 + v1)
        traj.append(x.clamp(-1, 1))

    return torch.stack(traj, dim=0)  # [steps+1, N, C, H, W]


def show_trajectory_collated(traj: torch.Tensor, nrow: Optional[int] = None, every: int = 1, title: Optional[str] = None):
    """
    Visualize a whole trajectory as a single image panel.

    Args
    ----
    traj  : [T, N, C, H, W] in [-1, 1]   (output of sample_flow_x_all)
    nrow  : grid columns for each time step (defaults to ~sqrt(N))
    every : subsample time steps (e.g., every=2 to show every other step)
    title : optional title

    Behavior
    --------
    For each time step t, builds a grid of the N samples with nrow columns.
    All grids are stacked vertically and shown as one image.
    """
    assert traj.dim() == 5, "traj must be [T, N, C, H, W]"
    T, N, C, H, W = traj.shape
    if nrow is None:
        nrow = int(math.sqrt(N)) or 1

    # build a grid for each time step (optionally subsampled)
    grids = []
    for t in range(0, T, every):
        g = make_grid(traj[t], nrow=nrow, padding=0)   # [-1, 1], shape [C, Ht, Wt]
        grids.append(g)

    # stack time-step grids vertically: [C, sum(Ht), Wt]
    panel = torch.cat(grids, dim=1)

    # to numpy image in [0,1]
    img = panel.detach().cpu().numpy().transpose(1, 2, 0).squeeze()
    img = (img + 1) / 2

    # figure size heuristic: width ~ nrow, height scales with number of time rows
    rows_per_grid = math.ceil(N / nrow)
    plt.figure(figsize=(nrow, max(1, rows_per_grid) * len(grids)))
    plt.imshow(img, cmap="gray" if C == 1 else None, interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_clean_collated(x1_list, num_digits: int = 25, nrow: Optional[int] = None, plot_show: bool = False, save_path: Optional[str] = None):
    """
    Display a collated grid of clean MNIST digits from a list of x1 tensors.
    
    Args:
        x1_list (list[Tensor]): List of tensors shaped [N, 1, H, W]
        num_digits (int): Total number of digits to plot
        nrow (int): Number of images per row (default: sqrt of num_digits)
        plot_show (bool): Whether to plt.show() the figure.
        save_path (str|None): Optional path to save the image.

    Notes:
        Handles inputs scaled either in [0,1] or [-1,1] by auto-rescaling.
    """
    # Concatenate all provided x1 batches into one tensor
    all_digits = torch.cat(x1_list, dim=0)

    # Limit to desired number of digits
    all_digits = all_digits[:num_digits]

    # Choose row size if not given
    if nrow is None:
        nrow = max(1, int(num_digits**0.5))

    # Normalize to [0,1] if necessary
    with torch.no_grad():
        mn, mx = float(all_digits.min()), float(all_digits.max())
    if mx <= 1.0 and mn >= 0.0:
        img_for_grid = all_digits
    else:
        img_for_grid = (all_digits + 1) / 2  # assume [-1,1]

    # Make a grid
    grid = make_grid(img_for_grid, nrow=nrow, padding=0)

    # Convert to numpy [0,1]
    np_img = grid.detach().cpu().permute(1, 2, 0).squeeze().numpy()

    plt.figure(figsize=(nrow, max(1, num_digits // nrow)))
    plt.imshow(np_img, cmap="gray")
    plt.axis("off")
    if save_path:
        dirpath = os.path.dirname(save_path) or "."
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if plot_show:
        plt.show()
    plt.close()



def show_collated(tensor_batch: torch.Tensor, nrow: Optional[int] = None, title: Optional[str] = None,
                  save_path: Optional[str] = None, plot_show: bool = False):
    """
    Display a collated grid for a batch.

    Parameters
    ----------
    tensor_batch : torch.Tensor
        Shape [N, C, H, W], expected in [-1,1] or [0,1] (auto-rescaled).
    nrow : Optional[int]
        Number of columns in the grid; defaults to ~sqrt(N).
    title : Optional[str]
        Matplotlib title.
    save_path : Optional[str]
        If provided, save figure to this path.
    plot_show : bool
        If True, call plt.show(). Always closes the figure at the end.
    """
    N, C = tensor_batch.size(0), tensor_batch.size(1)
    if nrow is None:
        nrow = int(math.sqrt(N)) or 1

    # Auto-rescale
    with torch.no_grad():
        mn, mx = float(tensor_batch.min()), float(tensor_batch.max())
    x = tensor_batch if (0.0 <= mn and mx <= 1.0) else (tensor_batch + 1) / 2

    grid = make_grid(x, nrow=nrow, padding=0)  # [0,1]
    img = grid.detach().cpu().numpy().transpose(1, 2, 0).squeeze()

    plt.figure(figsize=(nrow, max(1, N // nrow)))
    plt.imshow(img, cmap="gray" if C == 1 else None, interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    if save_path:
        dirpath = os.path.dirname(save_path) or "."
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if plot_show:
        plt.show()
    plt.close()


def save_trajectory_gif(traj: torch.Tensor, path: str, nrow: Optional[int] = None, every: int = 1, fps: int = 10,
                        times: Optional[torch.Tensor] = None, cmap_if_gray: str = "gray"):
    """
    Build a GIF from a trajectory [T, N, C, H, W].
    Title is drawn INSIDE the frame for reliable visibility across backends.

    Returns
    -------
    str : saved path
    """
    assert traj.dim() == 5, "traj must be [T, N, C, H, W]"
    T, N, C, H, W = traj.shape
    if nrow is None:
        nrow = int(math.sqrt(N)) or 1

    # time axis (defaults to [0,1])
    if times is None:
        times = torch.linspace(0, 1, T, device=traj.device)
    times = times.detach().cpu()[::every]

    # precompute per-frame grids in [0,1]
    frames = []
    for t in range(0, T, every):
        g = make_grid(traj[t], nrow=nrow, padding=0)          # [C, Ht, Wt], in [-1,1]
        img = g.detach().cpu().permute(1, 2, 0).squeeze()     # [Ht, Wt] or [Ht, Wt, C]
        img = (img + 1) / 2                                   # -> [0,1]
        frames.append(img)

    rows_per_grid = math.ceil(N / nrow)
    fig = plt.figure(figsize=(nrow, max(1, rows_per_grid)))
    ax = fig.add_axes([0, 0, 1, 1])  # full-frame axes
    ax.axis("off")

    first = frames[0]
    if (first.ndim == 2) or (C == 1):
        im = ax.imshow(first, cmap=cmap_if_gray, interpolation="nearest")
    else:
        im = ax.imshow(first, interpolation="nearest")

    # Draw time label INSIDE the axes (top-left), with a readable background
    txt = ax.text(
        0.01, 0.99, f"t = {float(times[0]):.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12, color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2")
    )

    def update(i):
        im.set_data(frames[i])
        txt.set_text(f"t = {float(times[i]):.3f}")
        return [im, txt]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    writer = animation.PillowWriter(fps=fps)
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    anim.save(path, writer=writer)
    plt.close(fig)
    return path




def save_trajectory_strip(
    traj: torch.Tensor,
    path: str,
    nrow: Optional[int] = None,
    every: int = 1,
    times: Optional[torch.Tensor] = None,
    cmap_if_gray: str = "gray",
    dpi: int = 200,
    pad_inches: float = 0.02,
    height: float = 3.0,  # figure height (inches) per frame; width scales automatically
):
    """
    Save a horizontal strip of frames from a trajectory [T, N, C, H, W]
    into a SINGLE figure (PNG/PDF/SVG... based on `path` extension).

    - For each time t, we first tile the N images into a grid (nrow x rows_per_grid),
      exactly like in your GIF function, then place those grids side-by-side.
    - Use `every` to subsample time steps (e.g., every=2).
    - If C==1, a grayscale colormap is used.

    Returns
    -------
    str : the saved path
    """
    assert traj.dim() == 5, "traj must be [T, N, C, H, W]"
    T, N, C, H, W = traj.shape
    if nrow is None:
        nrow = int(math.sqrt(N)) or 1

    # time axis (defaults to [0,1])
    if times is None:
        times = torch.linspace(0, 1, T, device=traj.device)
    times = times.detach().cpu()[::every]

    # Build per-time grids in [0,1]
    frames = []
    for t in range(0, T, every):
        g = make_grid(traj[t], nrow=nrow, padding=0)          # [C, Ht, Wt] in [-1,1]
        img = g.detach().cpu().permute(1, 2, 0).squeeze()     # [Ht, Wt] or [Ht, Wt, C]
        img = (img + 1) / 2                                   # -> [0,1]
        frames.append(img)

    if len(frames) == 0:
        raise ValueError("No frames selected (check `every`).")

    # Figure geometry
    first = frames[0]
    Ht, Wt = first.shape[:2] if first.ndim >= 2 else (1, 1)
    K = len(frames)
    aspect = Wt / float(Ht) if Ht else 1.0
    fig_height = max(1.0, float(height))
    fig_width = max(1.0, K * aspect * fig_height)

    fig, axes = plt.subplots(1, K, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]  # 1 x K

    for i, (ax, img) in enumerate(zip(axes, frames)):
        ax.axis("off")
        if (img.ndim == 2) or (C == 1):
            ax.imshow(img, cmap=cmap_if_gray, interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")

        ax.set_title(f"t = {float(times[i]):.3f}", fontsize=28)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.show()
    plt.close(fig)
    return path


def plot_msint_pairs_over_epochs(loader, val_loader=None, epochs=1, n_pairs=3, shuffle=False):
    """
    Plot the first n_pairs (x0, x1) pairs from loader and val_loader
    across several epochs to visualize reshuffling.
    """

    # epoch = 1
    # if shuffle:
    #     print("Shuffling pairings in the dataset (pre-training)")
    #     loader.shuffle_pairings(seed=0, epoch=epoch)
    #     print(loader.order0)

    #     if val_loader is not None:
    #         val_loader.shuffle_pairings(seed=0, epoch=epoch)

    for epoch in range(epochs):
        # Shuffle if supported (your CachedLoader has shuffle())
        if hasattr(loader, "shuffle"):
            loader.shuffle(seed=0, epoch=epoch)
        if val_loader is not None and hasattr(val_loader, "shuffle"):
            val_loader.shuffle(seed=0, epoch=epoch)

        # Extract first n_pairs from train loader
        x0s, x1s = [], []
        for i, (x0, x1) in enumerate(loader):
            x0s.append(x0[0].cpu())
            x1s.append(x1[0].cpu())
            if i >= n_pairs - 1:
                break

        # Plot train loader pairs
        fig, axes = plt.subplots(n_pairs, 2, figsize=(4, 2*n_pairs))
        fig.suptitle(f"Train loader – Epoch {epoch+1}", fontsize=14)
        for i in range(n_pairs):
            axes[i, 0].imshow(x0s[i].squeeze(), cmap="gray")
            axes[i, 0].set_title("x0")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(x1s[i].squeeze(), cmap="gray")
            axes[i, 1].set_title("x1")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

        # Do the same for val loader if provided
        if val_loader is not None:
            x0s, x1s = [], []
            for i, (x0, x1) in enumerate(val_loader):
                x0s.append(x0[0].cpu())
                x1s.append(x1[0].cpu())
                if i >= n_pairs - 1:
                    break

            fig, axes = plt.subplots(n_pairs, 2, figsize=(4, 2*n_pairs))
            fig.suptitle(f"Val loader – Epoch {epoch+1}", fontsize=14)
            for i in range(n_pairs):
                axes[i, 0].imshow(x0s[i].squeeze(), cmap="gray")
                axes[i, 0].set_title("x0")
                axes[i, 0].axis("off")
                axes[i, 1].imshow(x1s[i].squeeze(), cmap="gray")
                axes[i, 1].set_title("x1")
                axes[i, 1].axis("off")
            plt.tight_layout()
            plt.show()

