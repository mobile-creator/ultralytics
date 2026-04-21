# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth Anything V2 architecture modules: DINOv2 encoder + DPT decoder.

Exact reimplementation of:
  https://github.com/DepthAnything/Depth-Anything-V2/tree/main/depth_anything_v2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    """Residual convolution unit: relu → conv → relu → conv + skip."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Fuse path from deeper level with skip from current level, then upsample."""

    def __init__(self, features: int):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, *xs, size=None):
        """Forward pass.

        Args:
            xs: Either (path,) or (path, skip).
            size: Target spatial size for interpolation. If None, uses scale_factor=2.
        """
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res
        output = self.resConfUnit2(output)
        if size is not None:
            output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)
        else:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        output = self.out_conv(output)
        return output


class DPTHead(nn.Module):
    """Dense Prediction Transformer head for monocular depth estimation."""

    def __init__(self, in_channels: int, features: int = 256, out_channels: list[int] | None = None,
                 metric: bool = False):
        super().__init__()
        if out_channels is None:
            out_channels = [features // 4, features // 2, features, features]
        self.metric = metric

        # Reassemble: project encoder features to out_channels
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
            for out_ch in out_channels
        ])

        # Spatial resize to create multi-scale pyramid
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # scratch.layer{1-4}_rn: project to uniform feature dim
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(out_ch, features, kernel_size=3, stride=1, padding=1, bias=False)
            for out_ch in out_channels
        ])

        # scratch.refinenet{1-4}: feature fusion
        self.refinenets = nn.ModuleList([
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
            FeatureFusionBlock(features),
        ])

        # Output head
        head_features_1 = features
        head_features_2 = 32
        self.head_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        final_conv = nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0)
        if metric:
            # Unbounded logits for sigmoid(x)*max_depth mapping. Bias = -2 so sigmoid(-2) ≈ 0.12,
            # giving an initial output ≈ 0.12 * max_depth ≈ 1.2m for max_depth=10 — reasonable indoor start.
            nn.init.constant_(final_conv.bias, -2.0)
            self.head_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                final_conv,
            )
        else:
            self.head_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                final_conv,
                nn.ReLU(True),
            )

    def forward(self, patch_h: int, patch_w: int, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            patch_h: Height in patches.
            patch_w: Width in patches.
            features: List of 4 feature tensors from ViT, each (B, N, C).

        Returns:
            Depth tensor (B, 1, target_H, target_W) at input resolution.
        """
        out = []
        for i, x in enumerate(features):
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1_rn = self.layer_rn[0](out[0])
        layer_2_rn = self.layer_rn[1](out[1])
        layer_3_rn = self.layer_rn[2](out[2])
        layer_4_rn = self.layer_rn[3](out[3])

        # Bottom-up fusion with explicit target sizes
        path_4 = self.refinenets[3](layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.refinenets[2](path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.refinenets[1](path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.refinenets[0](path_2, layer_1_rn)

        # Output head: conv → upsample to full res → conv+relu+conv+relu
        out = self.head_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.head_conv2(out)

        return out


class DepthAnythingV2(nn.Module):
    """Depth Anything V2 model: DINOv2 encoder + DPT decoder."""

    CONFIGS = {
        "vits": {"encoder": "dinov2_vits14", "embed_dim": 384, "features": 64,
                 "out_channels": [48, 96, 192, 384], "intermediate_layers": [2, 5, 8, 11]},
        "vitb": {"encoder": "dinov2_vitb14", "embed_dim": 768, "features": 128,
                 "out_channels": [96, 192, 384, 768], "intermediate_layers": [2, 5, 8, 11]},
        "vitl": {"encoder": "dinov2_vitl14", "embed_dim": 1024, "features": 256,
                 "out_channels": [256, 512, 1024, 1024], "intermediate_layers": [4, 11, 17, 23]},
    }

    def __init__(self, encoder_name: str = "vits", metric: bool = False):
        super().__init__()
        cfg = self.CONFIGS[encoder_name]
        self.encoder_name = encoder_name
        self.intermediate_layers = cfg["intermediate_layers"]
        self.metric = metric

        self.encoder = torch.hub.load(
            "facebookresearch/dinov2", cfg["encoder"], pretrained=False
        )
        self.encoder.requires_grad_(False)

        self.depth_head = DPTHead(
            in_channels=cfg["embed_dim"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
            metric=metric,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RGB image → relative depth map.

        Args:
            x: Input tensor (B, 3, H, W), H and W must be divisible by 14.

        Returns:
            Depth map (B, H, W).
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        # get_intermediate_layers returns list of (features, cls_token) tuples
        features = self.encoder.get_intermediate_layers(
            x, n=self.intermediate_layers, return_class_token=True
        )
        # Extract just the patch features (not cls token)
        features = [f[0] for f in features]

        depth = self.depth_head(patch_h, patch_w, features)
        if not self.metric:
            # Zero-shot relative depth path: clamp to non-negative before LS alignment.
            depth = F.relu(depth)

        return depth.squeeze(1)  # (B, H, W)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, input_size: int = 518) -> torch.Tensor:
        """Resize to input_size, predict, resize back."""
        _, _, h, w = x.shape
        scale = input_size / max(h, w)
        new_h = max(14, (int(h * scale) // 14) * 14)
        new_w = max(14, (int(w * scale) // 14) * 14)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)
        depth = self.forward(x)
        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True)
        return depth.squeeze(1)
