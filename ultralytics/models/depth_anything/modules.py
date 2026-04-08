# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth Anything V2 architecture modules: DINOv2 encoder + DPT decoder.

Reference: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    """Residual convolution unit with two 3×3 convolutions and a residual connection."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Fuse a skip connection with the upsampled path via residual conv units."""

    def __init__(self, features: int):
        super().__init__()
        self.res_conv1 = ResidualConvUnit(features)
        self.res_conv2 = ResidualConvUnit(features)
        self.output_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            # Resize path to match skip spatial dims (handles odd patch sizes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = x + skip
        x = self.res_conv1(x)
        x = self.res_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.output_conv(x)
        return x


class DPTHead(nn.Module):
    """Dense Prediction Transformer head for monocular depth estimation.

    Fuses multi-scale features from a ViT encoder into a single-channel depth map
    via reassemble + fusion layers, following the DPT architecture used in
    Depth Anything V2.
    """

    def __init__(self, in_channels: int, features: int = 256, out_channels: list[int] | None = None):
        """Initialize DPT head.

        Args:
            in_channels: Number of channels from the ViT encoder (embed_dim).
            features: Internal feature dimension for fusion layers.
            out_channels: Per-layer output channels for reassemble projections.
        """
        super().__init__()
        if out_channels is None:
            out_channels = [features // 4, features // 2, features, features]

        # Step 1: Project from encoder embed_dim → out_channels[i] (1×1 conv)
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1, bias=True)
            for out_ch in out_channels
        ])

        # Step 2: Spatial resize to create multi-scale feature pyramid
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Step 3: Project from out_channels[i] → features (1×1 conv, "layer_rn")
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(out_ch, features, kernel_size=3, stride=1, padding=1, bias=False)
            for out_ch in out_channels
        ])

        # Step 4: Feature fusion refinement (bottom-up)
        self.refinenets = nn.ModuleList([
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
        ])

        # Step 5: Output head → single channel depth
        self.head_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.head_conv2 = nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1)
        self.head_conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, patch_h: int, patch_w: int, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass: fuse multi-scale ViT features into a depth map.

        Args:
            patch_h: Height in patches.
            patch_w: Width in patches.
            features: List of 4 feature tensors from ViT layers, each (B, N, C).

        Returns:
            Depth map tensor of shape (B, 1, H, W).
        """
        out = []
        for i, x in enumerate(features):
            # (B, N, C) → (B, C, patch_h, patch_w) → project → resize → layer_rn
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            x = self.layer_rn[i](x)
            out.append(x)

        # Bottom-up fusion (layer 3 → 0)
        path = self.refinenets[3](out[3])
        path = self.refinenets[2](path, out[2])
        path = self.refinenets[1](path, out[1])
        path = self.refinenets[0](path, out[0])

        # Output head
        out = self.head_conv1(path)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.head_conv2(out)
        out = F.relu(out, inplace=True)
        out = self.head_conv3(out)
        return F.relu(out, inplace=True)


class DepthAnythingV2(nn.Module):
    """Depth Anything V2 model: DINOv2 encoder + DPT decoder.

    Produces relative (affine-invariant) inverse depth from RGB images.
    """

    # Architecture configs keyed by encoder size
    CONFIGS = {
        "vits": {"encoder": "dinov2_vits14", "embed_dim": 384, "features": 64,
                 "out_channels": [48, 96, 192, 384], "intermediate_layers": [2, 5, 8, 11]},
        "vitb": {"encoder": "dinov2_vitb14", "embed_dim": 768, "features": 128,
                 "out_channels": [96, 192, 384, 768], "intermediate_layers": [2, 5, 8, 11]},
        "vitl": {"encoder": "dinov2_vitl14", "embed_dim": 1024, "features": 256,
                 "out_channels": [256, 512, 1024, 1024], "intermediate_layers": [4, 11, 17, 23]},
    }

    def __init__(self, encoder_name: str = "vits"):
        """Initialize Depth Anything V2.

        Args:
            encoder_name: One of 'vits', 'vitb', 'vitl'.
        """
        super().__init__()
        cfg = self.CONFIGS[encoder_name]
        self.encoder_name = encoder_name
        self.intermediate_layers = cfg["intermediate_layers"]

        # DINOv2 encoder (loaded via torchhub)
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2", cfg["encoder"], pretrained=False
        )
        self.encoder.requires_grad_(False)

        # DPT decoder
        self.depth_head = DPTHead(
            in_channels=cfg["embed_dim"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: RGB image → relative depth map.

        Args:
            x: Input RGB tensor (B, 3, H, W), H and W must be divisible by 14.

        Returns:
            Depth map tensor (B, H, W).
        """
        B, _, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14

        # Extract intermediate features from DINOv2
        features = self.encoder.get_intermediate_layers(
            x, n=self.intermediate_layers, reshape=False
        )

        # DPT fusion → depth
        depth = self.depth_head(patch_h, patch_w, list(features))
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        return depth.squeeze(1)  # (B, H, W)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, input_size: int = 518) -> torch.Tensor:
        """Convenience method: resize to input_size, predict, resize back."""
        _, _, h, w = x.shape
        scale = input_size / max(h, w)
        new_h = max(14, (int(h * scale) // 14) * 14)
        new_w = max(14, (int(w * scale) // 14) * 14)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

        depth = self.forward(x)

        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True)
        return depth.squeeze(1)
