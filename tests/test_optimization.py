"""Tests for optimization module."""

import pytest
import torch

from graviton_native.optimization import get_device
from graviton_native.optimization.device import is_mac_mps


def test_get_device_returns_device():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_is_mac_mps_returns_bool():
    assert isinstance(is_mac_mps(), bool)


def test_get_device_prefer_mps_false():
    dev = get_device(prefer_mps=False)
    # On Mac without MPS preference, might still get mps if cuda not available
    assert dev.type in ("cuda", "mps", "cpu")
