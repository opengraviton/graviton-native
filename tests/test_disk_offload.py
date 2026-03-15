"""Tests for disk-offload training (config and helpers, no full 72b run)."""

import tempfile
from pathlib import Path

import pytest

from graviton_native.training.disk_offload import (
    _get_disk_offload_config,
    train_72b_disk_offload,
)


def test_disk_offload_config_72b():
    config = _get_disk_offload_config("72b")
    assert config.num_hidden_layers == 80
    assert config.hidden_size == 8192


def test_disk_offload_config_36b():
    config = _get_disk_offload_config("36b")
    assert config.num_hidden_layers == 40
    assert config.hidden_size == 8192


def test_train_72b_disk_offload_alias():
    """train_72b_disk_offload is an alias for train_disk_offload(model_size='72b')."""
    assert callable(train_72b_disk_offload)
