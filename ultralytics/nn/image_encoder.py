# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Image encoder model and loss for universal encoder distillation.

Distill one or more frozen teachers (EUPE, DINOv3, SAM3, SigLIP2) into a YOLO backbone using
CLS + patch token feature matching. Supports single-teacher and multi-teacher distillation.

Loss formulation follows EUPE (arXiv:2603.22387) Eq.5-6 and AM-RADIO (arXiv:2312.06709) Eq.2-3:
  Per teacher: L_cls = cosine(student, teacher), L_patch = 0.9*cosine + 0.1*smooth_L1
  Multi-teacher: L = sum_i (L_cls_i + L_patch_i) -- EUPE Eq.6, equal weighting
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import ClassificationModel
from ultralytics.nn.teacher_model import safe_key


def _make_adaptor(in_dim, out_dim):
    """Create a 2-layer MLP adaptor head per EUPE Section 4.1.

    Architecture: "linear projection without bias, LayerNorm, GELU, linear without bias" (EUPE arXiv:2603.22387, Section
    4.1). EUPE uses hidden_dim=3072 for 86M+ students; we use in_dim (1280) proportionate to the 6.7M YOLO26s backbone.
    RADIO MLP v1 uses ReLU (verified RADIO/radio/adaptor_mlp.py:27); we follow EUPE's GELU.
    """
    return nn.Sequential(
        nn.Linear(in_dim, in_dim, bias=False),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim, bias=False),
    )


class ImageEncoderLoss:
    """Multi-teacher CLS + patch token distillation loss for encoder pretraining.

    Per-teacher loss (EUPE Eq.5): L_cls_i = 1 - cos_sim(student_cls, teacher_cls) L_patch_i = alpha * (1 - cos_sim(s,
    t)) + beta * smooth_L1(s, t) Multi-teacher total (EUPE Eq.6): L = sum_i (L_cls_i + L_patch_i) -- equal weighting, no
    teacher dropping

    Alpha=0.9, beta=0.1 per EUPE Eq.5. AM-RADIO (arXiv:2312.06709, Section 3.3) states "to mostly rely on the
    empirically better cosine distance, but also match vector magnitudes".

    Skip L_cls for patches-only teachers (SAM3, ConvNeXt) per DUNE token_types convention (verified:
    dune/teachers/config.py:25,36).

    Attributes:
        cos_weight (float): Alpha in EUPE Eq.5.
        l1_weight (float): Beta in EUPE Eq.5.
    """

    def __init__(self, cos_weight=0.9, l1_weight=0.1):
        """Initialize ImageEncoderLoss.

        Args:
            cos_weight (float): Alpha weight for cosine similarity in patch loss.
            l1_weight (float): Beta weight for smooth L1 in patch loss.
        """
        self.cos_weight = cos_weight
        self.l1_weight = l1_weight

    def _teacher_loss(self, s_cls, s_patch, t_cls, t_patch):
        """Compute loss for a single teacher (EUPE Eq.5).

        Args:
            s_cls (torch.Tensor): Student CLS features (B, D).
            s_patch (torch.Tensor): Student patch features (B, N, D).
            t_cls (torch.Tensor | None): Teacher CLS features or None for patches-only.
            t_patch (torch.Tensor): Teacher patch features (B, N, D).

        Returns:
            (tuple): (teacher_loss, [cls_cos, patch_cos, patch_l1]).
        """
        # Spatial alignment: if student and teacher have different patch counts, interpolate teacher
        # to match student grid. EUPE Section 3.1 upsamples to max(N_S, N_T); for teachers with
        # very large grids (SAM3: 5184 patches), we downsample teacher to student grid instead.
        # TODO: for SAM3, use pixel-shuffle upsampling in the adaptor MLP instead of downsampling
        # teacher patches. C-RADIOv4 (RADIO/radio/adaptor_mlp.py:99-107) uses einops.rearrange
        # with upsample_factor to produce higher-res student patches matching SAM's dense grid.
        t_patch = t_patch.to(s_patch)
        if t_patch.shape[1] != s_patch.shape[1]:
            h = int(s_patch.shape[1] ** 0.5)
            th = int(t_patch.shape[1] ** 0.5)
            t_patch = t_patch.transpose(1, 2).reshape(t_patch.shape[0], t_patch.shape[2], th, th)
            t_patch = F.interpolate(t_patch, size=(h, h), mode="bilinear", antialias=True)
            t_patch = t_patch.flatten(2).transpose(1, 2)

        # Force fp32 for loss: fp16 cosine_similarity eps=1e-8 rounds to 0, causing nan on
        # near-zero features (random init). Follows DUNE (dune/model/losses.py:58).
        with torch.autocast(device_type=s_cls.device.type, dtype=torch.float32, enabled=s_cls.is_cuda):
            # EUPE Eq.5 line 1: CLS cosine distance (skip for patches-only teachers)
            if t_cls is not None:
                cls_cos = 1.0 - F.cosine_similarity(s_cls, t_cls.to(s_cls), dim=-1).mean()
            else:
                cls_cos = torch.tensor(0.0, device=s_cls.device)
            # EUPE Eq.5 line 2: patch alpha*cosine + beta*smooth_L1
            patch_cos = 1.0 - F.cosine_similarity(s_patch, t_patch, dim=-1).mean()
            patch_l1 = F.smooth_l1_loss(s_patch, t_patch)
            loss = cls_cos + self.cos_weight * patch_cos + self.l1_weight * patch_l1

        return loss, [cls_cos.detach(), patch_cos.detach(), patch_l1.detach()]

    def __call__(self, preds, batch):
        """Compute multi-teacher distillation loss (EUPE Eq.6: sum over teachers).

        Args:
            preds (dict): {teacher_key: (student_cls, student_patches)} per teacher.
            batch (dict): {teacher_key: {"cls": Tensor|None, "patches": Tensor}} per teacher. Must also contain
                "_teacher_keys" listing the active teacher keys.

        Returns:
            (tuple): (total_loss, stacked loss_items for all teachers).
        """
        teacher_keys = batch["_teacher_keys"]
        total_loss = torch.tensor(0.0, device=next(iter(preds.values()))[0].device)
        all_items = []

        for key in teacher_keys:
            s_cls, s_patch = preds[key]
            t_cls = batch[key]["cls"]
            t_patch = batch[key]["patches"]
            loss_i, items_i = self._teacher_loss(s_cls, s_patch, t_cls, t_patch)
            total_loss = total_loss + loss_i
            all_items.extend(items_i)

        return total_loss, torch.stack(all_items)


class ImageEncoderModel(ClassificationModel):
    """YOLO backbone with per-teacher CLS and patch adaptor heads for encoder distillation.

    Architecture follows EUPE (arXiv:2603.22387) ConvNeXt student recipe:
    - CLS via global avg pool (verified eupe/models/convnext.py:220)
    - Patches via bilinear upsample to teacher grid (verified eupe/models/convnext.py:256)
    - Shared LayerNorm on concatenated [CLS; patches] before splitting (verified eupe/models/convnext.py:224)
    - Per-teacher 2-layer MLP adaptor heads for CLS and patches (AM-RADIO pattern: head_mlp + feat_mlp,
    verified RADIO/radio/adaptor_generic.py; EUPE Section 4.1 MLP spec)

    Multi-teacher: each teacher gets its own adaptor pair with its own embed_dim and grid size. Single-teacher mode is
    the special case with one entry in the adaptors dict.

    Attributes:
        token_norm (nn.LayerNorm): Shared norm for [CLS; patches] (EUPE ConvNeXt pattern).
        adaptors (nn.ModuleDict): Per-teacher adaptor heads {safe_name: ModuleDict{"cls": MLP, "patch": MLP}}.
        teacher_grids (dict): Per-teacher spatial grid height {safe_name: int}.
    """

    def __init__(self, cfg="yolo26s-cls.yaml", ch=3, nc=1000, verbose=True, teachers=None):
        """Initialize ImageEncoderModel with per-teacher adaptor heads.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int): Number of classes (unused during distillation).
            verbose (bool): Whether to display model information.
            teachers (dict): Per-teacher config. Keys are teacher names (e.g. "eupe:vitb16"), values are dicts with
                "embed_dim", "num_patches", "token_types". If None, defaults to a single EUPE-ViT-B teacher for
                backward compat.
        """
        super().__init__(cfg, ch, nc, verbose)
        if teachers is None:
            teachers = {"eupe:vitb16": {"embed_dim": 768, "num_patches": 256, "token_types": ("cls", "patches")}}

        c_ = self.model[-1].linear.in_features  # 1280

        # Shared LayerNorm on concatenated [CLS; patches] before adaptor heads
        # Matches EUPE ConvNeXt (verified eupe/models/convnext.py:224):
        #   x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))
        self.token_norm = nn.LayerNorm(c_)

        # Per-teacher adaptor heads (EUPE Stage 1: one adaptor pair per teacher, Eq.4-6)
        # AM-RADIO uses separate head_mlp + feat_mlp per teacher (verified RADIO/radio/adaptor_generic.py)
        self.adaptors = nn.ModuleDict()
        self.teacher_grids = {}
        for name, tcfg in teachers.items():
            safe = safe_key(name)
            heads = nn.ModuleDict()
            if "cls" in tcfg["token_types"]:
                heads["cls"] = _make_adaptor(c_, tcfg["embed_dim"])
            heads["patch"] = _make_adaptor(c_, tcfg["embed_dim"])
            self.adaptors[safe] = heads
            self.teacher_grids[safe] = int(tcfg["num_patches"] ** 0.5) if tcfg["num_patches"] > 0 else 16

    def loss(self, batch, preds=None):
        """Compute multi-teacher distillation loss from backbone features.

        Args:
            batch (dict): Batch with 'img' and per-teacher entries {key: {"cls": T|None, "patches": T}}.
            preds: Unused (computed internally).

        Returns:
            (tuple): (loss, loss_items) from ImageEncoderLoss.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        x = batch["img"]
        for m in self.model[:-1]:
            x = m(x)
        head = self.model[-1]
        features = head.conv(x)  # (B, 1280, H, W) shared features

        # CLS via global avg pool (EUPE ConvNeXt: x_pool = x.mean([-2, -1]))
        cls_feats = head.pool(features).flatten(1)  # (B, 1280)

        # Per-teacher: interpolate to teacher grid, normalize, project through adaptors.
        # EUPE convnext.py:224 does norm(cat([cls, patches])), but LayerNorm(c_) normalizes
        # over channel dim independently per position, so separate application is equivalent.
        cls_normed = self.token_norm(cls_feats)
        teacher_preds = {}
        for key in self.adaptors:
            h = self.teacher_grids[key]
            # Patches via bilinear upsample (EUPE convnext.py:256, bilinear+antialias)
            patch_feats = F.interpolate(features, size=(h, h), mode="bilinear", antialias=True)
            patch_feats = patch_feats.flatten(2).transpose(1, 2)  # (B, N, 1280)
            patch_normed = self.token_norm(patch_feats)

            heads = self.adaptors[key]
            s_cls = heads["cls"](cls_normed) if "cls" in heads else cls_normed
            s_patch = heads["patch"](patch_normed)
            teacher_preds[key] = (s_cls, s_patch)

        result = self.criterion(teacher_preds, batch)

        # Nan instrumentation: log diagnostic info on first nan occurrence per training run
        if result[1].isnan().any() and not getattr(self, "_nan_logged", False):
            self._nan_logged = True
            import logging

            log = logging.getLogger("ultralytics")
            log.warning("NAN DETECTED in loss items. Diagnostic dump:")
            log.warning(f"  loss_items: {result[1].tolist()}")
            log.warning(
                f"  features: nan={features.isnan().any()}, inf={features.isinf().any()}, "
                f"max={features.abs().max():.4f}, dtype={features.dtype}"
            )
            log.warning(f"  cls_normed: nan={cls_normed.isnan().any()}, max={cls_normed.abs().max():.4f}")
            for key in self.adaptors:
                s_cls, s_patch = teacher_preds[key]
                t_data = batch[key]
                log.warning(
                    f"  {key}/s_cls: nan={s_cls.isnan().any()}, inf={s_cls.isinf().any()}, "
                    f"max={s_cls.abs().max():.4f}, dtype={s_cls.dtype}"
                )
                log.warning(
                    f"  {key}/s_patch: nan={s_patch.isnan().any()}, inf={s_patch.isinf().any()}, "
                    f"max={s_patch.abs().max():.4f}"
                )
                log.warning(
                    f"  {key}/t_cls: nan={t_data['cls'].isnan().any() if t_data['cls'] is not None else 'N/A'}, "
                    f"max={t_data['cls'].abs().max():.4f if t_data['cls'] is not None else 'N/A'}"
                )
                log.warning(
                    f"  {key}/t_patch: nan={t_data['patches'].isnan().any()}, max={t_data['patches'].abs().max():.4f}"
                )
            # Check adaptor weight magnitudes
            for key in self.adaptors:
                for name, p in self.adaptors[key].named_parameters():
                    if p.isnan().any() or p.isinf().any():
                        log.warning(f"  PARAM {key}/{name}: nan={p.isnan().any()}, inf={p.isinf().any()}")
                    if p.abs().max() > 1000:
                        log.warning(f"  PARAM {key}/{name}: max={p.abs().max():.1f} (large)")

        return result

    def init_criterion(self):
        """Initialize distillation loss."""
        return ImageEncoderLoss()
