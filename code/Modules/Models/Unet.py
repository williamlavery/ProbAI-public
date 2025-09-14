#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVectorField-U: a compact U-Net for time-conditioned *flow matching* on images.

Version: 1.1.0
Date: 2025-08-26
Author: W. Lavery 
Python: >=3.9

------------------------------------------------------------------------------
What this file contains
------------------------------------------------------------------------------
• `time_features(t, H, W)`: lightweight Fourier/positional time embedding
  expanded spatially to H×W.
• Normalization helper `_gn(c)` producing a robust GroupNorm for any channel size.
• Encoder/decoder building blocks (`DoubleConv`, `Down`, `Up`) using GroupNorm+SiLU.
• `TinyVectorField_U`: a small U-Net that predicts a *time-conditional velocity
  field* v_θ(x_t, t) for Flow Matching.

------------------------------------------------------------------------------
The theory (how this supports Flow Matching)
------------------------------------------------------------------------------
Flow Matching (FM) learns a velocity field v_θ(x, t) that matches a target
field v*(x, t) along reference paths connecting source and data distributions.
In the trainer, with a *linear path* x(t) = (1−t)x0 + t x1, one trains with:

    L(θ) = E_{(x0,x1), t} || v_θ(x_t, t) − (x1 − x0) ||²,
    where x_t = (1−t)x0 + t x1 and t ~ Uniform[0, 1].

This module provides the *model side*:
  • A compact U-Net consumes x_t and a spatially-broadcast encoding of t.
  • Time is injected as feature maps so every pixel can modulate its behavior
    across t (helps when velocity magnitude/structure varies over time).
  • The output is a single-channel per-pixel velocity component (for grayscale
    problems). For multi-channel data, adapt `in_ch`/`outc` accordingly.

Noisy path variant (context for future extensions):
  If training uses Gaussian perturbations around the path mean μ_t with σ(t),
  FM’s conditional target becomes
      v*(x̃_t, t) = (x1 − x0) + (d/dt log σ(t)) · (x̃_t − μ_t).
  The architecture here remains applicable; only the training target changes.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_features(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Create spatial time-conditioning maps from scalar times.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B, 1, 1, 1], values in [0, 1]. Typically sampled per-batch elem.
    H : int
        Target spatial height for broadcasting.
    W : int
        Target spatial width for broadcasting.

    Returns
    -------
    torch.Tensor
        Tensor of shape [B, C_t, H, W], where
        C_t = 1 + 2 * len(freqs). Contains the raw t plus sinusoidal
        features sin(2π f t) and cos(2π f t) for multiple frequencies.

    Notes
    -----
    • Using a small bank of sinusoidal features gives the network smooth
      access to different time scales without learning them from scratch.
    • Features are broadcast (tiled) over H×W so each pixel receives the
      same time embedding, preserving spatial alignment.
    """
    """
    t: [B,1,1,1] in [0,1]
    returns: [B, C_t, H, W] concatenated sin/cos features + raw t
    """
    freqs = [1, 2, 4, 8, 16]
    feats = [t]
    for f in freqs:
        w = 2 * math.pi * f
        feats.append(torch.sin(w * t))
        feats.append(torch.cos(w * t))
    f = torch.cat(feats, dim=1)  # [B, 1 + 2*len(freqs), 1, 1]
    return f.expand(-1, -1, H, W)  # tile over HxW


# ---------------- Normalization helper ----------------
def _gn(c: int) -> nn.GroupNorm:
    """
    GroupNorm factory:
      - Use the largest group count among {8, 4, 2, 1} that divides `c`.
      - Falls back to 1 group (LayerNorm across channels) if nothing else divides.

    Why GroupNorm?
      • BatchNorm depends on batch statistics → noisy with small batches (common in generative training).
      • InstanceNorm is too restrictive (no cross-channel interaction).
      • GroupNorm strikes a balance: stable with small batch sizes, 
        allows limited cross-channel normalization, and is widely used in diffusion models.
    """
    for g in [8, 4, 2, 1]:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


# ---------------- Encoder/Decoder building blocks ----------------
class DoubleConv(nn.Module):
    """
    Two Conv2d + GroupNorm + SiLU activations in sequence.

    Motivation:
      • Two stacked 3×3 convolutions (instead of one) increase effective receptive field
        without aggressive downsampling.
      • Each conv is followed by GroupNorm and SiLU:
          - GroupNorm: batch-size agnostic normalization, stabilizes training.
          - SiLU (a.k.a. Swish): smooth, non-saturating, empirically stronger than ReLU in diffusion/flow nets.
      • This block is the "workhorse" of both encoder and decoder.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _gn(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _gn(out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two 3×3 convs each followed by GroupNorm and SiLU."""
        return self.block(x)


class Down(nn.Module):
    """
    Downsample block: AvgPool2d → DoubleConv.

    Motivation:
      • Downsampling halves spatial size (H,W) and doubles channels,
        trading spatial resolution for more feature capacity.
      • Average pooling avoids checkerboard artifacts compared to stride-2 convs.
      • Followed by DoubleConv to process compressed representation.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by 2 with average pooling, then refine via DoubleConv."""
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsample block: Bilinear Upsample → Concatenate skip connection → DoubleConv.

    Motivation:
      • Upsample with bilinear interpolation: smoother, fewer checkerboard artifacts 
        than transposed convolutions.
      • Concatenate encoder features ("skip connections") to restore high-frequency
        spatial detail lost during downsampling.
      • DoubleConv after concatenation fuses skip and upsampled paths.
      • Includes padding logic to handle odd input sizes (ensures dimensions match).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Features from the previous (coarser) stage.
        skip : torch.Tensor
            Skip-connection features from the encoder with matching spatial size.

        Returns
        -------
        torch.Tensor
            Fused features after upsampling and DoubleConv.
        """
        x = self.up(x)

        # Align shapes if odd-sized inputs caused a mismatch
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, (diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2))

        # Channel concat: [up_path, skip_path]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TinyVectorField_U(nn.Module):
    """
    UNet-based predictor for v(x_t, t) with input [x_t, time_features(t)].

    Design intent (Flow Matching context):
      • Predict a time-conditioned vector field v(x_t, t) with a *small* U-Net:
        enough capacity for 2D digits/patches, but shallow to keep training stable.
      • Condition on time via *spatial* feature maps (broadcasted over H×W)
        so the network can modulate features locally at every pixel.
      • Use skip connections to preserve high-frequency details of x_t,
        which are crucial when predicting velocity fields.
    """
    def __init__(self, Tfeats: int = 11, width: int = 64):
        """
        Parameters
        ----------
        Tfeats : int, default 11
            Number of time-feature channels (raw t + sine/cosine pairs).
            A compact set (≈8–16) balances expressivity and efficiency.
        width : int, default 64
            Base channel count. Channels double with depth (U-Net convention).
            64 is suitable for ~28–64 px grayscale images.

        Notes
        -----
        • Input channels = 1 (grayscale x_t) + Tfeats.
        • Output channels = 1 (per-pixel velocity component). For multi-channel
          targets, change the head to match dimensionality.
        """
        super().__init__()
        self.Tfeats = Tfeats

        # Input channels = 1 (grayscale x_t) + time features.
        # Motivation: concatenate conditioning to let early layers "see" t directly,
        # rather than injecting t later via FiLM/affine layers (keeps code simple,
        # works well for small models).
        in_ch = 1 + self.Tfeats
        self.in_ch = in_ch

        # ---------------- Encoder ----------------
        # Each stage roughly: 2×(Conv→Norm→Act), then spatial downsample.
        # Motivation: double conv improves receptive field & nonlinearity per scale
        # without increasing downsampling depth. Downsampling increases channels,
        # trading spatial size for feature richness.
        self.inc   = DoubleConv(in_ch, width)            #  H×W,     C:    1+T -> 64
        self.down1 = Down(width, width * 2)              #  H/2×W/2, C:   64 -> 128
        self.down2 = Down(width * 2, width * 4)          #  H/4×W/4, C:  128 -> 256

        # ---------------- Bottleneck ----------------
        # Keep same channels at the smallest scale for compute stability; adding more
        # depth here is possible but unnecessary for small images.
        self.bot = DoubleConv(width * 4, width * 4)

        # ---------------- Decoder ----------------
        # Up blocks: upsample + concat skip + double conv.
        # Concatenation doubles channels (skip + upsampled path), hence (in = up_C + skip_C).
        # Motivation: skip connections restore spatial detail lost during downsampling.
        self.up2 = Up(width * 4 + width * 4, width * 2)  # concat with down2 skip (256+256 -> 128)
        self.up1 = Up(width * 2 + width * 2, width)      # concat with down1 skip (128+128 -> 64)
        self.up0 = Up(width + width, width)              # concat with inc    skip  (64+64   -> 64)

        # ---------------- Head ----------------
        # Final 3×3 conv to map features → 1 channel velocity field component per pixel.
        # Padding=1 keeps spatial size. No activation: v(x_t, t) is unconstrained.
        self.outc = nn.Conv2d(width, 1, kernel_size=3, padding=1)

        # Initialize the last layer to zero so early training starts near v≈0.
        # Motivation: stabilizes optimization in flow/score-style training where
        # target magnitudes can vary with t; lets earlier blocks learn before the head
        # imposes large outputs.
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_t : torch.Tensor
            Shape [B, 1, H, W]. Current state on the path (or noisy variant).
        t : torch.Tensor
            Shape [B, 1, 1, 1]. Scalar times in [0, 1], broadcastable spatially.

        Returns
        -------
        torch.Tensor
            Predicted velocity field v_θ(x_t, t), shape [B, 1, H, W].

        Flow Matching motivation for conditioning:
          Convert t → multi-channel spatial features so every pixel’s prediction
          can depend on t with local context (simple and effective for 2D images).
        """
        # Convert scalar t to spatial maps of shape [B, Tfeats, H, W].
        # Typically implemented with Fourier/time embeddings then broadcast.
        tf = time_features(t, x_t.shape[-2], x_t.shape[-1])

        # Channel-wise concat so the network sees x_t and t side-by-side.
        x = torch.cat([x_t, tf], dim=1)  # [B, 1+Tfeats, H, W]

        # ---------------- U-Net core ----------------
        # Encoder path with skip saves features at each scale for later fusion.
        x1 = self.inc(x)    # shallow spatial details
        x2 = self.down1(x1) # mid-level features
        x3 = self.down2(x2) # coarse, semantic features

        # Bottleneck mixes global context at the coarsest resolution.
        xb = self.bot(x3)

        # Decoder: upsample and fuse with corresponding encoder activations.
        x = self.up2(xb, x3)
        x = self.up1(x,  x2)
        x = self.up0(x,  x1)

        # Linear head → predicted vector field v(x_t, t).
        return self.outc(x)
