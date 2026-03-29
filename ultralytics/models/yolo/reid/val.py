# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import ReidDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, TQDM
from ultralytics.utils.metrics import ReidMetrics
from ultralytics.utils.plotting import plot_images


class ReidValidator(BaseValidator):
    """Validator for person re-identification models.

    Accumulates embeddings, person IDs, and camera IDs during validation, then computes
    mAP and CMC metrics using the standard Market-1501 protocol.

    Attributes:
        query_feats (list): Accumulated query feature embeddings.
        query_pids (list): Accumulated query person IDs.
        query_camids (list): Accumulated query camera IDs.
        gallery_feats (list): Accumulated gallery feature embeddings.
        gallery_pids (list): Accumulated gallery person IDs.
        gallery_camids (list): Accumulated gallery camera IDs.
        metrics (ReidMetrics): Metrics calculator.

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidValidator
        >>> args = dict(model="yolo26n-reid.pt", data="Market-1501.yaml")
        >>> validator = ReidValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize ReidValidator.

        Args:
            dataloader: DataLoader for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Validation configuration.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "reid"
        self.metrics = ReidMetrics()
        self._feats = []
        self._pids = []
        self._camids = []

    def get_desc(self) -> str:
        """Return formatted description string."""
        return ("%22s" + "%11s" * 4) % ("", "mAP", "Rank-1", "Rank-5", "Rank-10")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize tracking containers."""
        self._model = model  # store reference for gallery feature extraction
        self.names = model.names
        self.nc = len(model.names)
        self._feats = []
        self._pids = []
        self._camids = []

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch."""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def update_metrics(self, preds, batch: dict[str, Any]) -> None:
        """Accumulate embeddings and metadata.

        When reid_tta is enabled, also forwards horizontally-flipped images through the model
        and averages the two embeddings for improved accuracy.

        Args:
            preds: Model output (embedding, feat_bn) tuple or just embedding.
            batch (dict): Batch with 'cls' and 'camid' keys.
        """
        emb = preds[0] if isinstance(preds, (list, tuple)) else preds

        # Flip TTA: average original + horizontally flipped embeddings
        if getattr(self.args, "reid_tta", False):
            with torch.no_grad():
                preds_flip = self._model(batch["img"].flip(dims=[3]))
            emb_flip = preds_flip[0] if isinstance(preds_flip, (list, tuple)) else preds_flip
            emb = (emb + emb_flip) / 2

        self._feats.append(emb.cpu())
        self._pids.append(batch["cls"].cpu())
        self._camids.append(torch.tensor([batch["camid"][i] for i in range(len(batch["camid"]))]) if isinstance(batch["camid"], list) else batch["camid"].cpu())

    def postprocess(self, preds):
        """Extract primary prediction from model output."""
        return preds

    def finalize_metrics(self) -> None:
        """Finalize metrics with speed info."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, float]:
        """Compute mAP and CMC from accumulated features.

        This performs the full query-vs-gallery evaluation. The dataloader iterates over
        the query split. For gallery evaluation, we build a separate dataset.
        """
        if not self._feats:
            return self.metrics.results_dict

        # Current accumulated features are from the val split (query)
        query_feats = torch.cat(self._feats, dim=0).numpy()
        query_pids = torch.cat(self._pids, dim=0).numpy()
        query_camids = torch.cat(self._camids, dim=0).numpy()

        # Build gallery dataset and extract features
        data = self.data
        gallery_path = data.get("gallery", data.get("test", ""))
        if not gallery_path:
            LOGGER.warning("No gallery path found in dataset config. Using query as gallery.")
            gallery_feats, gallery_pids, gallery_camids = query_feats, query_pids, query_camids
        else:
            if not Path(gallery_path).is_absolute():
                gallery_path = str(Path(data["path"]) / gallery_path)
            gallery_feats, gallery_pids, gallery_camids = self._extract_gallery_features(gallery_path)

        reranking = getattr(self.args, "reid_reranking", False)
        tag = " (re-ranking)" if reranking else ""
        LOGGER.info(f"{'Computing metrics':>22s}   {len(query_pids)} query x {len(gallery_pids)} gallery{tag} ...")
        self.metrics.process(query_feats, query_pids, query_camids,
                            gallery_feats, gallery_pids, gallery_camids,
                            reranking=reranking)
        return self.metrics.results_dict

    def _extract_gallery_features(self, gallery_path: str):
        """Extract features from gallery set.

        Args:
            gallery_path (str): Path to gallery images.

        Returns:
            Tuple of (features, pids, camids) as numpy arrays.
        """
        dataset = ReidDataset(root=gallery_path, args=self.args, augment=False, prefix="gallery", data=self.data)
        loader = build_dataloader(dataset, self.args.batch, self.args.workers, rank=-1)

        tta = getattr(self.args, "reid_tta", False)
        feats, pids, camids = [], [], []
        bar = TQDM(loader, desc=f"{'Extracting gallery':>22s}", total=len(loader))
        for batch in bar:
            batch["img"] = batch["img"].to(self.device, non_blocking=True)
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()

            with torch.no_grad():
                preds = self._model(batch["img"])
            emb = preds[0] if isinstance(preds, (list, tuple)) else preds

            if tta:
                with torch.no_grad():
                    preds_flip = self._model(batch["img"].flip(dims=[3]))
                emb_flip = preds_flip[0] if isinstance(preds_flip, (list, tuple)) else preds_flip
                emb = (emb + emb_flip) / 2

            feats.append(emb.cpu())
            pids.append(batch["cls"])
            camids.append(
                torch.tensor([batch["camid"][i] for i in range(len(batch["camid"]))]) if isinstance(batch["camid"], list) else batch["camid"]
            )

        return (
            torch.cat(feats, dim=0).numpy(),
            torch.cat(pids, dim=0).numpy(),
            torch.cat(camids, dim=0).numpy(),
        )

    def gather_stats(self) -> None:
        """Gather stats from all GPUs for DDP."""
        if RANK == 0:
            gathered_feats = [None] * dist.get_world_size()
            gathered_pids = [None] * dist.get_world_size()
            gathered_camids = [None] * dist.get_world_size()
            dist.gather_object(self._feats, gathered_feats, dst=0)
            dist.gather_object(self._pids, gathered_pids, dst=0)
            dist.gather_object(self._camids, gathered_camids, dst=0)
            self._feats = [f for rank in gathered_feats for f in rank]
            self._pids = [p for rank in gathered_pids for p in rank]
            self._camids = [c for rank in gathered_camids for c in rank]
        elif RANK > 0:
            dist.gather_object(self._feats, None, dst=0)
            dist.gather_object(self._pids, None, dst=0)
            dist.gather_object(self._camids, None, dst=0)

    def build_dataset(self, img_path: str) -> ReidDataset:
        """Create a ReidDataset instance for validation."""
        return ReidDataset(root=img_path, args=self.args, augment=False, prefix="query", data=self.data)

    def get_dataloader(self, dataset_path: str | Path, batch_size: int) -> torch.utils.data.DataLoader:
        """Build dataloader for validation.

        Args:
            dataset_path (str | Path): Path to dataset.
            batch_size (int): Batch size.

        Returns:
            (DataLoader): Validation dataloader.
        """
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self) -> None:
        """Print evaluation metrics."""
        pf = "%22s" + "%11.4g" * 4
        LOGGER.info(pf % ("Results", self.metrics.mAP, self.metrics.rank1, self.metrics.rank5, self.metrics.rank10))

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation (query) samples with person ID labels."""
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])
        plot_images(
            labels=batch,
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch: dict[str, Any], preds, ni: int) -> None:
        """Plot predictions (no-op for ReID, embeddings are not visual)."""
        pass
