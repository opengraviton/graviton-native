"""Tests for training pipeline."""

import tempfile
from pathlib import Path

import pytest

from graviton_native.training.trainer import get_preset_config


def test_get_preset_config_350m():
    config = get_preset_config("350m")
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 24
    assert config.vocab_size == 32000


def test_get_preset_config_1b():
    config = get_preset_config("1b")
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 24


def test_get_preset_config_72b():
    config = get_preset_config("72b")
    assert config.hidden_size == 8192
    assert config.num_hidden_layers == 80
    assert config.rope_theta == 1000000.0


def test_get_preset_config_unknown_defaults_to_350m():
    config = get_preset_config("unknown")
    assert config.hidden_size == 1024
