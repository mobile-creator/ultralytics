# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, IS_DEBIAN_TRIXIE, LOGGER
from ultralytics.utils.checks import check_apt_requirements, is_sudo_available

from .base import BaseBackend


class DeepXBackend(BaseBackend):
    """DeepX NPU inference backend for DeepX hardware accelerators.

    Loads compiled DeepX models (.dxnn files) and runs inference using the DeepX dx_engine runtime. Requires the
    dx_engine package to be installed.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a DeepX model from a directory containing a .dxnn file.

        Args:
            weight (str | Path): Path to the DeepX model directory containing the .dxnn binary.

        Raises:
            FileNotFoundError: If no .dxnn file is found in the given directory.
        """
        cmd = ["dxrt-cli", "--version"]
        help_url = "https://github.com/sixfab/sixfab_dx/"
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            LOGGER.info(f"\nDeepX inference requires the DeepX runtime. Attempting install from {help_url}")
            if not (IS_DEBIAN_TRIXIE and ARM64):
                raise OSError("DeepX runtime auto-install is only supported on Debian Trixie (arm64).")
            sudo = "sudo " if is_sudo_available() else ""
            for c in (
                f"wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg | {sudo}gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg",
                f'echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" | {sudo}tee /etc/apt/sources.list.d/sixfab-dx.list',
            ):
                subprocess.run(c, shell=True, check=True, stdout=subprocess.DEVNULL)
            check_apt_requirements(["sixfab-dx"])

        try:
            from dx_engine import InferenceEngine
        except ImportError:
            if IS_DEBIAN_TRIXIE and ARM64:
                wheels = sorted(Path("/opt/sixfab-dx/wheels").glob("dx_engine-*.whl"))
                if not wheels:
                    raise FileNotFoundError(
                        "No dx_engine wheel found in /opt/sixfab-dx/wheels/. Ensure sixfab-dx is installed."
                    )
                subprocess.run(["pip", "install", str(wheels[-1])], check=True)
            else:
                raise OSError(
                    "dx_engine is not installed. Auto-install is only supported on Debian Trixie (arm64). "
                    "Please install dx_engine manually and try again."
                )
            from dx_engine import InferenceEngine

        ver = (
            subprocess.run(cmd, capture_output=True, check=True).stdout.decode().splitlines()[0].split()[-1].lstrip("v")
        )
        LOGGER.info(f"Loading {weight} for DeepX inference... (runtime v{ver})")

        w = Path(weight)
        found = next(w.rglob("*.dxnn"), None)
        if found is None:
            raise FileNotFoundError(f"No .dxnn file found in: {w}")

        self.model = InferenceEngine(str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run inference on the DeepX NPU.

        Converts each image from BCHW float [0, 1] to HWC uint8 [0, 255] per the DeepX runtime contract,
        runs the engine per image, then stacks outputs along the batch dimension.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
        """
        outputs = []
        for sample in im.cpu().numpy():
            sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
            for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
                if i == len(outputs):
                    outputs.append([])
                outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
        y = [np.concatenate(x, axis=0) for x in outputs]
        return y[0] if len(y) == 1 else y
