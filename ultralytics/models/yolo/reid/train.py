# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import ReidDataset, build_reid_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ReidModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first


class ReidTrainer(BaseTrainer):
    """Trainer for person re-identification models.

    Extends BaseTrainer with ReID-specific dataset handling (Market-1501), PK batch sampling,
    and multi-loss training (cross-entropy + triplet).

    Attributes:
        model (ReidModel): The ReID model to be trained.
        data (dict): Dataset information including identity names and count.
        loss_names (list[str]): Names of loss components: ['ce_loss', 'tri_loss'].

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidTrainer
        >>> args = dict(model="yolo26n-reid.yaml", data="Market-1501.yaml", epochs=60)
        >>> trainer = ReidTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize ReidTrainer.

        Args:
            cfg (dict): Default configuration dictionary.
            overrides (dict, optional): Parameter overrides.
            _callbacks (list, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "reid"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 256
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the model's identity names and configure loss from trainer args."""
        nc = self.data["nc"]
        self.model.names = {i: str(i) for i in range(nc)}
        # Configure loss criterion with trainer args
        from ultralytics.utils.loss import ReIDLoss

        self.model.criterion = ReIDLoss(
            nc=self.data["nc"],
            triplet_margin=getattr(self.args, "triplet_margin", 0.3),
            label_smooth=getattr(self.args, "label_smoothing", 0.1),
            triplet_weight=getattr(self.args, "triplet_weight", 1.0),
            ce_weight=getattr(self.args, "ce_weight", 1.0),
            center_weight=getattr(self.args, "center_weight", 0.0),
            center_momentum=getattr(self.args, "center_momentum", 0.9),
            focal_gamma=getattr(self.args, "focal_gamma", 0.0),
            supcon_temp=getattr(self.args, "supcon_temp", 0.0),
        )

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a ReidModel configured for training.

        Args:
            cfg: Model configuration.
            weights: Pre-trained weights.
            verbose (bool): Whether to display model info.

        Returns:
            (ReidModel): Configured model.
        """
        model = ReidModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def setup_model(self):
        """Load or create model for ReID tasks."""
        ckpt = super().setup_model()
        ReidModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Create a ReidDataset instance.

        Args:
            img_path (str): Path to dataset split.
            mode (str): 'train', 'val', or 'test'.
            batch: Unused.

        Returns:
            (ReidDataset): Dataset for the specified split.
        """
        return ReidDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode, data=self.data)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return dataloader with PK sampling for training.

        Args:
            dataset_path (str): Path to dataset.
            batch_size (int): Batch size.
            rank (int): Process rank for DDP.
            mode (str): 'train' or 'val'.

        Returns:
            (DataLoader): Configured dataloader.
        """
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        if mode == "train":
            # PK sampling: P identities x K images
            p = getattr(self.args, "reid_p", 16)
            k = getattr(self.args, "reid_k", 4)
            loader = build_reid_dataloader(
                dataset, batch_size, self.args.workers, p=p, k=k, shuffle=True, rank=rank
            )
        else:
            from ultralytics.data import build_dataloader

            loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)

        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess a batch of images and identity labels."""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def progress_string(self) -> str:
        """Return a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Return a ReidValidator instance."""
        self.loss_names = ["ce_loss", "tri_loss"]
        return yolo.reid.ReidValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items.

        ReID validation uses query-gallery mAP, not loss computation, so val-prefixed
        loss items are omitted to keep results.csv and plots clean.

        Args:
            loss_items: Loss tensor items.
            prefix (str): Prefix for loss names.

        Returns:
            Loss keys or dict of loss items.
        """
        if prefix == "val":
            return [] if loss_items is None else {}
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))

    def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
        """Plot training samples with annotations.

        Args:
            batch (dict): Batch with images and labels.
            ni (int): Batch iteration number.
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
