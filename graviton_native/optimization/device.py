"""Device selection optimized for Mac (MPS) and NVIDIA."""

from __future__ import annotations

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the best available device for training.
    On Mac: MPS (Metal) when available.
    On NVIDIA: CUDA.
    Fallback: CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    return torch.device("cpu")


def is_mac_mps() -> bool:
    """Check if running on Apple Silicon with MPS."""
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
