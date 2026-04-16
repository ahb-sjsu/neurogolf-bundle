"""Unified ONNX operator grammar for NeuroGolf (Phase 1, Week 1)."""
from .builder import GraphBuilder
from .primitives import (
    # Color-only
    identity_network,
    color_remap_network,
    # Spatial (static Gather indices)
    gather_network,
    gather_with_mask_network,
    # Local (Conv)
    single_conv_network,
    two_layer_conv_network,
    # Affine (learned spatial transform)
    affine_gather_network,
    # Utilities
    score_model,
    verify_model,
)

__all__ = [
    'GraphBuilder',
    'identity_network', 'color_remap_network',
    'gather_network', 'gather_with_mask_network',
    'single_conv_network', 'two_layer_conv_network',
    'affine_gather_network',
    'score_model', 'verify_model',
]
