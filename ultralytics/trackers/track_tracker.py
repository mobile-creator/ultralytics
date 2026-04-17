# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""TrackTrack: Track-focused online multi-object tracker.

Reference: Shim et al., "Focusing on Tracks for Online Multi-Object Tracking" (CVPR 2025).

Two additions on top of standard tracking-by-detection:
    - Track-Perspective-Based Association (TPA): multi-cue cost (HMIoU + cosine ReID + confidence
      projection + corner angle distance), solved with iterative assignment that progressively
      relaxes the matching threshold.
    - Track-Aware Initialization (TAI): spawns new tracks only after suppressing candidates that
      heavily overlap with existing tracks or higher-scoring detections.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import scipy.linalg
import torch

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .byte_tracker import BYTETracker, STrack
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH

# Corner index arrays for angle-distance vectorization: LT, LB, RT, RB of an (x1,y1,x2,y2) box
_CORNER_DX_IDX = np.array([0, 0, 2, 2])
_CORNER_DY_IDX = np.array([1, 3, 1, 3])


def _nsa_kalman_update(
    kf: KalmanFilterXYWH, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """NSA-Kalman update (StrongSORT): scale the measurement noise R by (1 - confidence).

    High-confidence detections pull the state estimate harder toward the observation; low-
    confidence ones have less influence. Only R is scaled, not the projected state covariance
    H P H^T, so we recompute the projection locally instead of calling `kf.project`.

    Confidence is clamped to avoid collapsing R to zero (would break Cholesky).
    """
    w = max(1.0 - float(confidence), 0.05)
    std = kf._std_weight_position * mean[2:4]
    R = np.diag(np.square(np.r_[std, std])) * w
    H = kf._update_mat
    projected_mean = H @ mean
    projected_cov = np.linalg.multi_dot((H, covariance, H.T)) + R

    chol, low = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
    gain = scipy.linalg.cho_solve((chol, low), np.dot(covariance, H.T).T, check_finite=False).T
    innovation = measurement - projected_mean
    new_mean = mean + innovation @ gain.T
    new_cov = covariance - np.linalg.multi_dot((gain, projected_cov, gain.T))
    return new_mean, new_cov


def _bbox_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two (N,4) and (M,4) box sets in xyxy format; returns (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _hmiou_distance(a_tracks: list, b_tracks: list) -> tuple[np.ndarray, np.ndarray]:
    """HMIoU = HIoU * IoU, where HIoU is vertical-overlap / vertical-union. Returns (iou_sim, 1-HMIoU).

    The height modifier improves matching when detections with similar aspect ratios are stacked
    vertically (common for pedestrians).
    """
    n, m = len(a_tracks), len(b_tracks)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64), np.ones((n, m), dtype=np.float64)
    a = np.ascontiguousarray([t.xyxy for t in a_tracks], dtype=np.float64)
    b = np.ascontiguousarray([t.xyxy for t in b_tracks], dtype=np.float64)
    iou_sim = _bbox_overlaps(a, b)
    h_over = np.minimum(a[:, 3:4], b[:, 3:4].T) - np.maximum(a[:, 1:2], b[:, 1:2].T)
    h_union = np.maximum(a[:, 3:4], b[:, 3:4].T) - np.minimum(a[:, 1:2], b[:, 1:2].T)
    h_iou = np.clip(h_over / (h_union + 1e-9), 0, 1)
    return iou_sim, 1.0 - h_iou * iou_sim


def _angle_distance(tracks: list, dets: list, frame_id: int, delta_t: int = 3) -> np.ndarray:
    """Angle distance between each track's corner velocities and the track-to-detection direction.

    For each (track, detection) pair the per-corner direction is computed from the track's box at
    frame_id - delta_t to the detection, then compared via arccos of the dot product with the
    track's stored corner velocity. Averaged over 4 corners and weighted by detection score.
    """
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.ones((n, m), dtype=np.float64)
    t_box = np.stack([t.get_history_box(frame_id, delta_t) for t in tracks])  # (N, 4)
    d_box = np.stack([d.xyxy for d in dets])  # (M, 4)
    deltas = d_box[None] - t_box[:, None]  # (N, M, 4)
    dx = deltas[:, :, _CORNER_DX_IDX]
    dy = deltas[:, :, _CORNER_DY_IDX]
    norms = np.sqrt(dx * dx + dy * dy) + 1e-5
    dx /= norms
    dy /= norms
    tv = np.stack([t.velocity for t in tracks])  # (N, 4, 2)
    dot = tv[:, None, :, 0] * dx + tv[:, None, :, 1] * dy
    dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi  # (N, M)
    return dist * np.array([d.score for d in dets])[None]


def _confidence_distance(tracks: list, dets: list) -> np.ndarray:
    """Absolute difference between each track's projected score and each detection's confidence."""
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)
    t_prev = np.array([t.prev_score for t in tracks])
    t_cur = np.array([t.score for t in tracks])
    t_proj = t_cur + (t_cur - t_prev)  # first-order extrapolation
    d_score = np.array([d.score for d in dets])
    return np.abs(t_proj[:, None] - d_score[None])


def _iterative_associate(cost: np.ndarray, match_thr: float, reduce_step: float = 0.05):
    """Greedy mutually-nearest matching with a threshold that shrinks each iteration.

    Returns (matches, unmatched_tracks, unmatched_dets).
    """
    matches = []
    cost = cost.copy()
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        md = np.argmin(cost, axis=1)
        mt = np.argmin(cost, axis=0)
        new = [[ti, md[ti]] for ti in range(cost.shape[0]) if mt[md[ti]] == ti and cost[ti, md[ti]] < match_thr]
        if not new:
            break
        matches.extend(new)
        for t, d in new:
            cost[t, :] = 1.0
            cost[:, d] = 1.0
        match_thr -= reduce_step
    m_t = {t for t, _ in matches}
    m_d = {d for _, d in matches}
    u_t = [i for i in range(cost.shape[0]) if i not in m_t]
    u_d = [i for i in range(cost.shape[1]) if i not in m_d]
    return matches, u_t, u_d


def _track_aware_nms(tracks: list, dets: list, tai_thr: float, init_thr: float) -> list[bool]:
    """TAI NMS: suppress detections that heavily overlap an existing track or a stronger detection."""
    if not dets:
        return []
    scores = np.array([d.score for d in dets])
    allow = list(scores > init_thr)
    if len(tracks) + len(dets) < 2:
        return allow
    boxes = np.ascontiguousarray([o.xyxy for o in tracks + dets], dtype=np.float64)
    iou = _bbox_overlaps(boxes, boxes)
    nt = len(tracks)
    for i in range(len(dets)):
        if not allow[i]:
            continue
        if nt and np.max(iou[nt + i, :nt]) > tai_thr:
            allow[i] = False
            continue
        for j in range(len(dets)):
            if i != j and allow[j] and scores[i] > scores[j] and iou[nt + i, nt + j] > tai_thr:
                allow[j] = False
    return allow


def attach_raw_preds_hook(predictor) -> None:
    """Wrap `predictor.postprocess` so raw pre-NMS predictions are captured for D_del.

    TrackTrack needs the pre-NMS model output to run the looser secondary NMS pass (paper Eq. 1).
    Idempotent. After wrapping, `_raw_preds` and `_preproc_img_shape` are populated each frame.
    """
    if hasattr(predictor, "_orig_postprocess"):
        return
    orig = predictor.postprocess

    @wraps(orig)
    def _wrapped(preds, img, *args, **kwargs):
        predictor._raw_preds = preds.clone() if isinstance(preds, torch.Tensor) else preds
        predictor._preproc_img_shape = img.shape[2:]
        return orig(preds, img, *args, **kwargs)

    predictor._orig_postprocess = orig
    predictor.postprocess = _wrapped


def compute_dets_del(predictor) -> list | None:
    """Run a loose IoU=0.95 NMS on the captured raw preds and return detections the tight NMS dropped.

    Returns per-batch `(xywh, conf, cls)` tuples (or None where no deleted detections are found),
    or None if no raw preds were captured. Consumes `_raw_preds` (clears it on the predictor).
    """
    raw = getattr(predictor, "_raw_preds", None)
    if raw is None:
        return None
    from torchvision.ops import box_iou

    from ultralytics.utils import nms, ops

    is_obb = predictor.args.task == "obb"
    preds_loose = nms.non_max_suppression(
        raw,
        predictor.args.conf,
        0.95,
        predictor.args.classes,
        predictor.args.agnostic_nms,
        max_det=predictor.args.max_det,
        nc=0 if predictor.args.task == "detect" else len(predictor.model.names),
        end2end=getattr(predictor.model, "end2end", False),
        rotated=is_obb,
    )

    im_shape = getattr(predictor, "_preproc_img_shape", None)
    out = []
    for loose, result in zip(preds_loose, predictor.results):
        tight = (result.obb if is_obb else result.boxes).cpu()
        if len(loose) == 0 or len(tight) == 0:
            out.append(None)
            continue
        loose = loose.clone()
        if im_shape is not None:
            loose[:, :4] = ops.scale_boxes(im_shape, loose[:, :4], result.orig_shape)
        tight_xyxy = tight.xyxy.cpu()
        loose_xyxy = loose[:, :4].cpu()
        if tight_xyxy.numel() == 0 or loose_xyxy.numel() == 0:
            out.append(None)
            continue
        max_iou = box_iou(loose_xyxy, tight_xyxy).max(dim=1).values
        mask = max_iou < 0.97
        if not mask.any():
            out.append(None)
            continue
        dels = loose[mask]
        xywh = ops.xyxy2xywh(dels[:, :4]).numpy()
        out.append((xywh, dels[:, 4].numpy(), dels[:, 5].numpy()))

    predictor._raw_preds = None
    return out


class TTSTrack(BaseTrack):
    """Single-object track for TrackTrack with corner velocity, score history, and ReID features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): Shared Kalman filter used for batch prediction.
        min_track_len (int): Class-level default; overridden by TRACKTRACK from config.
        kalman_filter (KalmanFilterXYWH): Per-track Kalman filter used after activation.
        mean (np.ndarray): Mean state vector.
        covariance (np.ndarray): Covariance matrix.
        score (float): Current detection confidence.
        prev_score (float): Confidence from the previous update (for score projection).
        tracklet_len (int): Successful updates since activation.
        velocity (np.ndarray): Per-corner (4,2) unit velocity vectors.
        smooth_feat (np.ndarray | None): EMA-smoothed ReID embedding.
        curr_feat (np.ndarray | None): Raw ReID embedding from the current frame.

    Examples:
        Create and activate a new track
        >>> track = TTSTrack([100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(KalmanFilterXYWH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYWH()
    min_track_len = 3  # overridden by TRACKTRACK.__init__
    _alpha = 0.95  # EMA smoothing for ReID features
    _delta_t = 3  # corner-velocity look-back window

    # STrack and TTSTrack both hold an 8-d Kalman state ordered (*box, *box_velocity) with the
    # position as the first two dims, so the GMC warp code works for both.
    multi_gmc = staticmethod(STrack.multi_gmc)

    def __init__(self, xywh: list[float], score: float, cls: Any, feat: np.ndarray | None = None):
        """Create a new track from a detection bounding box.

        Args:
            xywh: (x, y, w, h, idx) or (x, y, w, h, angle, idx), center-based with detection index.
            score: Detection confidence.
            cls: Class label.
            feat: Optional ReID feature vector.
        """
        super().__init__()
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter: KalmanFilterXYWH | None = None
        self.mean = self.covariance = None
        self.is_activated = False

        self.score = self.prev_score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

        self.velocity = np.zeros((4, 2), dtype=np.float64)
        self._history: dict[int, np.ndarray] = {}  # frame_id -> xyxy
        self.smooth_feat = self.curr_feat = None
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray) -> None:
        """Normalize `feat` and blend it into `smooth_feat` via score-adaptive EMA."""
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            beta = self._alpha + (1 - self._alpha) * (1 - self.score)
            self.smooth_feat = beta * self.smooth_feat + (1 - beta) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-9

    def get_history_box(self, frame_id: int, dt: int) -> np.ndarray:
        """Return the box from `dt` frames back; falls back to the most recent or current box."""
        target = frame_id - dt
        if target in self._history:
            return self._history[target].copy()
        if self._history:
            return self._history[max(self._history)].copy()
        return self.xyxy.copy()

    def predict(self) -> None:
        """Kalman predict for a single track."""
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[6] = mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[TTSTrack]) -> None:
        """Batched Kalman predict over a list of tracks."""
        if not stracks:
            return
        mm = np.asarray([s.mean.copy() for s in stracks])
        mc = np.asarray([s.covariance for s in stracks])
        for i, s in enumerate(stracks):
            if s.state != TrackState.Tracked:
                mm[i][6] = mm[i][7] = 0
        mm, mc = TTSTrack.shared_kalman.multi_predict(mm, mc)
        for i, (m, c) in enumerate(zip(mm, mc)):
            stracks[i].mean, stracks[i].covariance = m, c

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int) -> None:
        """Initialize Kalman state and promote to New state."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history[frame_id] = self.xyxy.copy()
        self.tracklet_len = 0
        self.state = TrackState.New
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new_track: TTSTrack, frame_id: int, new_id: bool = False) -> None:
        """Rebind a lost track to a fresh detection via NSA-Kalman."""
        self.prev_score = self.score
        self.mean, self.covariance = _nsa_kalman_update(
            self.kalman_filter, self.mean, self.covariance, self.convert_coords(new_track.tlwh), new_track.score
        )
        self._history[frame_id] = self.xyxy.copy()
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score, self.cls, self.angle, self.idx = new_track.score, new_track.cls, new_track.angle, new_track.idx

    def update(self, new_track: TTSTrack, frame_id: int) -> None:
        """Update a matched track with a new detection; promote to Tracked after min_track_len."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score
        self.mean, self.covariance = _nsa_kalman_update(
            self.kalman_filter, self.mean, self.covariance, self.convert_coords(new_track.tlwh), new_track.score
        )
        self._history[frame_id] = new_track.xyxy.copy()

        # Corner velocity: average unit-direction from each of the last delta_t box positions
        vel = np.zeros((4, 2), dtype=np.float64)
        cur = new_track.xyxy
        for dt in range(1, self._delta_t + 1):
            delta = cur - self.get_history_box(frame_id, dt)
            dx, dy = delta[_CORNER_DX_IDX], delta[_CORNER_DY_IDX]
            n = np.sqrt(dx * dx + dy * dy) + 1e-5
            vel += np.stack([dx / n, dy / n], axis=-1) / dt
        self.velocity = vel / self._delta_t

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        if self.state == TrackState.Tracked or self.tracklet_len >= self.min_track_len:
            self.state = TrackState.Tracked
            self.is_activated = True
        self.score, self.cls, self.angle, self.idx = new_track.score, new_track.cls, new_track.angle, new_track.idx

    @staticmethod
    def convert_coords(tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh to center-xywh for the Kalman filter."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        """Get (top-left x, top-left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Get (min x, min y, max x, max y)."""
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get (center x, center y, width, height)."""
        if self.mean is None:
            ret = self._tlwh.copy()
            ret[:2] += ret[2:] / 2
            return ret
        return self.mean[:4].copy()

    @property
    def xywha(self) -> np.ndarray:
        """Get (center x, center y, width, height, angle); falls back to xywh if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Packed tracking result: [*coords, track_id, score, cls, idx]."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Short string representation of the track."""
        return f"TT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class TRACKTRACK:
    """TrackTrack: multi-object tracker based on Track-Perspective Association and Track-Aware Init.

    Implements the full algorithm from Shim et al. (CVPR 2025): partitions detections into high,
    low, and deleted (loose-NMS recovered) sets; builds a multi-cue cost matrix (HMIoU + cosine
    ReID + confidence + angle); solves with iterative assignment; then runs TAI NMS on leftover
    high-confidence detections before spawning new tracks.

    Examples:
        Initialize and run on a single frame
        >>> tracker = TRACKTRACK(args, frame_rate=30)
        >>> tracked_objects = tracker.update(yolo_results, img=image)
    """

    # Reuse BYTETracker's purely-structural helpers (they only rely on .track_id / .frame_id /
    # .start_frame / .xyxy which TTSTrack exposes).
    joint_stracks = staticmethod(BYTETracker.joint_stracks)
    sub_stracks = staticmethod(BYTETracker.sub_stracks)
    remove_duplicate_stracks = staticmethod(BYTETracker.remove_duplicate_stracks)

    def __init__(self, args, frame_rate: int = 30):
        """Initialize from a tracker config. See `ultralytics/cfg/trackers/tracktrack.yaml`."""
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilterXYWH()
        TTSTrack.reset_id()

        # Association + cost-matrix weights
        g = lambda k, d: getattr(args, k, d)  # noqa: E731 - local getter for brevity
        self.det_thr = g("det_thr", 0.6)
        self.match_thr = g("match_thresh", 0.7)
        self.penalty_p = g("penalty_p", 0.2)
        self.penalty_q = g("penalty_q", 0.4)
        self.reduce_step = g("reduce_step", 0.05)
        self.iou_weight = g("iou_weight", 0.5)
        self.reid_weight = g("reid_weight", 0.5)
        self.conf_weight = g("conf_weight", 0.1)
        self.angle_weight = g("angle_weight", 0.05)
        # TAI
        self.tai_thr = g("tai_thr", 0.55)
        self.init_thr = g("init_thr", 0.7)
        TTSTrack.min_track_len = g("min_track_len", 3)

        # GMC (speed-tunable)
        self.gmc = GMC(method=g("gmc_method", "sparseOptFlow"), downscale=g("gmc_downscale", 3))
        if self.gmc.method == "sparseOptFlow":
            self.gmc.feature_params["maxCorners"] = g("gmc_max_corners", 200)
        self._gmc_skip = g("gmc_skip_frames", 0)
        self._gmc_warp = np.eye(2, 3, dtype=np.float64)
        self._gmc_counter = 0

        # ReID: "auto" reads backbone features; else path loads an external YOLO model
        self.with_reid = g("with_reid", False)
        self.encoder = None
        if self.with_reid:
            model = g("model", "auto")
            if model == "auto":
                self.encoder = lambda feats, _: [f.cpu().numpy() for f in feats]
            else:
                from .bot_sort import ReID

                self.encoder = ReID(model)

    def _cost_matrix(self, tracks: list[TTSTrack], dets: list[TTSTrack]) -> tuple[np.ndarray, np.ndarray]:
        """Build the full multi-cue cost matrix. Returns (iou_sim, cost)."""
        iou_sim, hmiou_dist = _hmiou_distance(tracks, dets)
        cost = self.iou_weight * hmiou_dist
        if self.with_reid and self.encoder is not None:
            cost += self.reid_weight * _cosine_distance(tracks, dets)
        else:
            cost += self.reid_weight * hmiou_dist  # fallback
        cost += self.conf_weight * _confidence_distance(tracks, dets)
        cost += self.angle_weight * _angle_distance(tracks, dets, self.frame_id)
        if iou_sim.size > 0:
            cost[iou_sim <= 0.10] = 1.0
        return iou_sim, np.clip(cost, 0, 1)

    def _apply_gmc(self, img: np.ndarray, dets_high: list, pools: list[list[TTSTrack]]) -> None:
        """Apply global motion compensation to `pools`, optionally reusing the cached warp."""
        if self._gmc_skip > 0 and self._gmc_counter % (self._gmc_skip + 1) != 0:
            warp = self._gmc_warp
        else:
            warp = self.gmc.apply(img, [t.xyxy for t in dets_high])
            self._gmc_warp = warp
        self._gmc_counter += 1
        for p in pools:
            TTSTrack.multi_gmc(p, warp)

    def update(
        self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, dets_del=None
    ) -> np.ndarray:
        """Advance one frame. Returns (N, 8) array: `[x1, y1, x2, y2, id, score, cls, idx]`."""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        # Partition detections by confidence
        scores = results.conf
        boxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        boxes = np.concatenate([boxes, np.arange(len(boxes)).reshape(-1, 1)], axis=-1)
        hi = scores >= self.args.track_high_thresh
        lo = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        # Build detection objects; attach ReID features to high-conf dets when enabled
        bh, sh, ch = boxes[hi], scores[hi], results.cls[hi]
        if self.with_reid and self.encoder is not None and img is not None and len(bh) > 0:
            features = self.encoder(img, bh)
            dets_high = [TTSTrack(b, s, c, f) for b, s, c, f in zip(bh, sh, ch, features)]
        else:
            dets_high = [TTSTrack(b, s, c) for b, s, c in zip(bh, sh, ch)]
        dets_low = [TTSTrack(b, s, c) for b, s, c in zip(boxes[lo], scores[lo], results.cls[lo])]

        # D_del: detections the tight NMS suppressed but looser NMS recovered (paper Eq. 1)
        dets_del_high: list[TTSTrack] = []
        if dets_del is not None:
            dx, dc, dcl = dets_del
            mask = dc > self.det_thr
            if mask.any():
                db = np.concatenate([dx[mask], -np.ones((mask.sum(), 1))], axis=-1)
                dets_del_high = [TTSTrack(b, s, c) for b, s, c in zip(db, dc[mask], dcl[mask])]

        # Split existing tracks into confirmed vs unconfirmed
        unconfirmed, tracked = [], []
        for t in self.tracked_stracks:
            (unconfirmed if not t.is_activated else tracked).append(t)
        pool = self.joint_stracks(tracked, self.lost_stracks)

        # GMC + Kalman predict
        if img is not None:
            self._apply_gmc(img, dets_high, [pool, unconfirmed])
        TTSTrack.multi_predict(pool)

        # Main association: confirmed pool vs (D_high + D_low + D_del)
        all_dets = dets_high + dets_low + dets_del_high
        n_high, n_low = len(dets_high), len(dets_low)
        _, cost = self._cost_matrix(pool, all_dets)
        if cost.shape[1] > n_high:
            cost[:, n_high : n_high + n_low] += self.penalty_p  # τ_p for D_low
        if dets_del_high:
            cost[:, n_high + n_low :] += self.penalty_q  # τ_q for D_del
        cost = np.clip(cost, 0, 1)

        matches, u_t, u_d = _iterative_associate(cost, self.match_thr, self.reduce_step)
        for ti, di in matches:
            t, d = pool[ti], all_dets[di]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id)
                activated.append(t)
            else:
                t.re_activate(d, self.frame_id, new_id=False)
                refind.append(t)
        for ti in u_t:
            t = pool[ti]
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost.append(t)

        # Second association: unconfirmed vs leftover high-conf detections (same multi-cue cost)
        leftover_high = [all_dets[i] for i in u_d if i < n_high]
        if unconfirmed and leftover_high:
            _, uc = self._cost_matrix(unconfirmed, leftover_high)
            uc_matches, uc_u_t, uc_u_d = _iterative_associate(uc, self.match_thr, self.reduce_step)
            for ti, di in uc_matches:
                unconfirmed[ti].update(leftover_high[di], self.frame_id)
                activated.append(unconfirmed[ti])
            for ti in uc_u_t:
                unconfirmed[ti].mark_removed()
                removed.append(unconfirmed[ti])
            leftover_high = [leftover_high[i] for i in uc_u_d]
        else:
            for t in unconfirmed:
                t.mark_removed()
                removed.append(t)

        # TAI: spawn new tracks only from candidates that survive NMS against existing tracks
        active = [t for t in self.tracked_stracks if t.state == TrackState.Tracked] + activated
        for det, ok in zip(leftover_high, _track_aware_nms(active, leftover_high, self.tai_thr, self.init_thr)):
            if ok:
                det.activate(self.kalman_filter, self.frame_id)
                activated.append(det)

        # Drop lost tracks past the buffer
        for t in self.lost_stracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed.append(t)

        # Bookkeeping
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray(
            [t.result for t in self.tracked_stracks if t.is_activated and t.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def reset(self) -> None:
        """Clear all tracker state including GMC warp history and the global ID counter."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        TTSTrack.reset_id()
        self.gmc.reset_params()


def _cosine_distance(tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
    """Cosine distance between each track's smoothed embedding and each detection's current feat.

    Missing embeddings are replaced with zero vectors so the pair contributes a constant distance.
    """
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.ones((n, m), dtype=np.float64)
    dim = 128  # fallback; overridden once any embedding is populated
    for obj in (*tracks, *dets):
        f = obj.smooth_feat if obj.smooth_feat is not None else obj.curr_feat
        if f is not None:
            dim = f.shape[0]
            break
    t_feat = np.stack([t.smooth_feat if t.smooth_feat is not None else np.zeros(dim, dtype=np.float32) for t in tracks])
    d_feat = np.stack([d.curr_feat if d.curr_feat is not None else np.zeros(dim, dtype=np.float32) for d in dets])
    return np.clip(1 - t_feat @ d_feat.T, 0, 1)
