"""
Mac-optimized training utilities.

- get_device(): Best device for current hardware (MPS on Apple Silicon)
- compile_layer(): Optional torch.compile for layer (when beneficial)
"""

from graviton_native.optimization.device import get_device

__all__ = ["get_device"]
