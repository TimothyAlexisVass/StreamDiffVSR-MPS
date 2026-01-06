"""
NeuFlow v2 wrapper for Stream-DiffVSR
Drop-in replacement for RAFT optical flow model.

NeuFlow v2 is 10-70x faster than RAFT while maintaining comparable accuracy.
Designed for edge devices and optimized for Apple Silicon via CoreML conversion.
"""

import torch
import torch.nn as nn
import sys
import os

# Add NeuFlow to path
NEUFLOW_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'neuflow_v2')
if NEUFLOW_PATH not in sys.path:
    sys.path.insert(0, NEUFLOW_PATH)

from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers for faster inference."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class NeuFlowWrapper(nn.Module):
    """
    Wrapper for NeuFlow v2 that provides RAFT-compatible interface.

    RAFT interface:
        flows = model(img1, img2)  # Returns list of flows
        flow = flows[-1]           # Final flow [B, 2, H, W]

    NeuFlow interface:
        flow_list = model(img0, img1)  # Returns list of flows
        flow = flow_list[-1]           # Final flow [B, 2, H, W]

    Both are compatible, but NeuFlow:
    1. Needs init_bhwd() called for specific dimensions
    2. Expects input in 0-255 range (normalized internally)
    3. Is 10-70x faster than RAFT
    """

    def __init__(self, weights_path=None, fuse_bn=True):
        super().__init__()

        self.model = NeuFlow()
        self._initialized_dims = None
        self._fused = False

        # Load weights if provided
        if weights_path is None:
            weights_path = os.path.join(NEUFLOW_PATH, 'neuflow_mixed.pth')

        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'], strict=True)
            print(f"Loaded NeuFlow weights from {weights_path}")
        else:
            print(f"Warning: NeuFlow weights not found at {weights_path}")

        self._fuse_bn = fuse_bn

    def _fuse_batchnorm(self):
        """Fuse conv and batchnorm layers for faster inference."""
        if self._fused:
            return

        for m in self.model.modules():
            if type(m) is ConvBlock:
                if hasattr(m, 'norm1') and hasattr(m, 'norm2'):
                    m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
                    m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
                    delattr(m, "norm1")
                    delattr(m, "norm2")
                    m.forward = m.forward_fuse

        self._fused = True
        print("NeuFlow: BatchNorm layers fused for faster inference")

    def _ensure_initialized(self, batch_size, height, width, device):
        """Initialize internal buffers for given dimensions."""
        dims = (batch_size, height, width, str(device))

        if self._initialized_dims != dims:
            # Determine if we should use amp (half precision)
            # MPS doesn't fully support half, so use float32
            use_amp = device.type == 'cuda'

            self.model.init_bhwd(batch_size, height, width, device, amp=use_amp)
            self._initialized_dims = dims

    def to(self, device):
        """Override to handle device transfer and re-initialization."""
        self._initialized_dims = None  # Force re-init on next forward
        return super().to(device)

    def forward(self, img1, img2):
        """
        Compute optical flow from img1 to img2.

        Args:
            img1: Source image [B, C, H, W], values in [0, 1] (will be scaled to 0-255)
            img2: Target image [B, C, H, W], values in [0, 1] (will be scaled to 0-255)

        Returns:
            List of flow tensors at different scales.
            Use flows[-1] for final flow [B, 2, H, W].
        """
        # Fuse batchnorm on first forward pass
        if self._fuse_bn and not self._fused:
            self._fuse_batchnorm()

        device = img1.device
        B, C, H, W = img1.shape

        # Initialize for current dimensions
        self._ensure_initialized(B, H, W, device)

        # NeuFlow expects 0-255 range, RAFT typically gets normalized images
        # Check if input is already in 0-255 range or needs scaling
        if img1.max() <= 1.0:
            img1 = img1 * 255.0
            img2 = img2 * 255.0

        # NeuFlow forward
        flow_list = self.model(img1, img2)

        return flow_list


def load_neuflow(device, weights_path=None):
    """
    Load NeuFlow v2 model as a drop-in replacement for RAFT.

    Args:
        device: torch.device to load model on
        weights_path: Optional path to weights file.
                     Defaults to neuflow_v2/neuflow_mixed.pth

    Returns:
        NeuFlowWrapper model ready for inference

    Usage:
        # Replace this:
        # of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

        # With this:
        of_model = load_neuflow(device)
    """
    model = NeuFlowWrapper(weights_path=weights_path)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    return model
