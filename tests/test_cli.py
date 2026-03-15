"""Tests for CLI."""

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from graviton_native.cli import _has_cuda, _is_mac_mps, main


def test_is_mac_mps_returns_bool():
    result = _is_mac_mps()
    assert isinstance(result, bool)


def test_has_cuda_returns_bool():
    result = _has_cuda()
    assert isinstance(result, bool)


def test_cli_help():
    with patch.object(sys, "argv", ["graviton-train", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_cli_run_help():
    with patch.object(sys, "argv", ["graviton-train", "run", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
