
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np

cv2.setNumThreads(0)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FACE_SIZE = 256

# ArcFace 5-point template (112x112) scaled to FACE_SIZE.
_ARCFACE_112 = np.array([
    [38.2946, 51.6963],   # left eye centre
    [73.5318, 51.5014],   # right eye centre
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)
TEMPLATE = _ARCFACE_112 * (FACE_SIZE / 112.0)

# MediaPipe FaceMesh landmark index sets.
LM_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LM_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LM_LEFT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LM_RIGHT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
LM_NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 278, 344, 45, 115, 48, 64, 220, 440]
LM_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
            0, 37, 39, 40, 185, 76, 306]
LM_CHIN = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 58]
LM_FOREHEAD = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 63, 105, 66, 107, 336, 296, 334, 293]
LM_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                234, 127, 162, 21, 54, 103, 67, 109]

# 5 alignment anchors: eye centres are means of the eye contours; rest are single points.
LM_NOSE_TIP = 1
LM_MOUTH_L = 61
LM_MOUTH_R = 291

# Generic 3D head model for solvePnP (nose tip, chin, eye corners, mouth corners).
_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
_PNP_IDS = [1, 152, 33, 263, 61, 291]

REGION_ORDER = ["forehead", "left_eye", "right_eye", "nose", "mouth", "chin",
                "left_cheek", "right_cheek"]
REGION_COLORS = {
    "forehead": (255, 128, 0), "left_eye": (0, 200, 255), "right_eye": (0, 140, 255),
    "nose": (0, 255, 120), "mouth": (200, 0, 255), "chin": (255, 0, 80),
    "left_cheek": (120, 120, 255), "right_cheek": (80, 80, 200),
}


# ---------------------------------------------------------------------------
# 1. Input
# ---------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(input_path, max_images=200, frame_stride=None):
    """Returns (list of BGR frames, list of human-readable source labels)."""
    if os.path.isdir(input_path):
        paths = sorted(p for p in glob.glob(os.path.join(input_path, "*"))
                       if os.path.splitext(p)[1].lower() in IMAGE_EXT)
        if not paths:
            raise FileNotFoundError(f"No images found in {input_path}")
        frames, labels = [], []
        for p in paths[:max_images]:
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
                labels.append(os.path.basename(p))
        print(f"Loaded {len(frames)} images from folder.")
        return frames, labels

    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXT:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = frame_stride or max(1, total // max_images if total else 1)
        frames, labels, idx = [], [], 0
        while len(frames) < max_images:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(frame)
                labels.append(f"frame_{idx:06d}")
            idx += 1
        cap.release()
        print(f"Extracted {len(frames)} frames (stride={stride}, total={total}).")
        return frames, labels

    raise ValueError(f"Unrecognised input: {input_path} (expected a folder or a video file)")


# ---------------------------------------------------------------------------
# 2. Detection + landmarks + alignment
# ---------------------------------------------------------------------------

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


def _ensure_landmarker_model(path=None):
    """Tasks API needs a .task bundle on disk. Download once, cache next to the script."""
    import urllib.request
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "face_landmarker.task")
    if not os.path.exists(path):
        print(f"Downloading face_landmarker.task -> {path}")
        urllib.request.urlretrieve(LANDMARKER_URL, path)
    return path


class LandmarkDetector:
    """Face landmarks with three backends, tried in order:

      1. mediapipe legacy  `mp.solutions.face_mesh`   (mediapipe <= ~0.10.14)
      2. mediapipe Tasks   `vision.FaceLandmarker`    (mediapipe >= 0.10.15; the legacy
                                                       solutions module was removed, so
                                                       this is the only path on new builds)
      3. OpenCV Haar cascade — box only, no real landmarks, approximate regions.

    Backend 2 also returns blendshapes, which we use for expression clustering.
    """

    def __init__(self, min_conf=0.3, model_path=None):
        self.mesh = None
        self.tasks = None
        self.backend = "haar"
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=True, min_detection_confidence=min_conf)
                self.backend = "mediapipe_facemesh"
            else:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                self._mp = mp
                opts = vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=_ensure_landmarker_model(model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    output_face_blendshapes=True,
                    min_face_detection_confidence=min_conf,
                    min_face_presence_confidence=min_conf)
                self.tasks = vision.FaceLandmarker.create_from_options(opts)
                self.backend = "mediapipe_tasks"
        except Exception as e:
            print(f"[warn] MediaPipe landmarks unavailable ({type(e).__name__}: {e}).\n"
                  f"       Falling back to Haar cascade — alignment and regions will be "
                  f"approximate. Fix with:  pip install -U mediapipe")
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def __call__(self, img_bgr):
        """Returns (landmarks Nx2 in pixel coords, blendshape vector or None)."""
        h, w = img_bgr.shape[:2]
        if self.mesh is not None:
            res = self.mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                return None, None
            lm = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark],
                          dtype=np.float32)
            return lm, None

        if self.tasks is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self.tasks.detect(mp_img)
            if not res.face_landmarks:
                return None, None
            lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
            bs = None
            if getattr(res, "face_blendshapes", None):
                bs = np.array([c.score for c in res.face_blendshapes[0]], dtype=np.float32)
            return lm, bs

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None, None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Synthesise 5 anchors from the box so the rest of the pipeline still works.
        lm = np.array([
            [fx + 0.32 * fw, fy + 0.40 * fh], [fx + 0.68 * fw, fy + 0.40 * fh],
            [fx + 0.50 * fw, fy + 0.58 * fh],
            [fx + 0.36 * fw, fy + 0.76 * fh], [fx + 0.64 * fw, fy + 0.76 * fh],
        ], dtype=np.float32)
        return lm, None


def five_anchors(lm):
    if len(lm) >= 468:
        return np.array([
            lm[LM_LEFT_EYE].mean(axis=0),
            lm[LM_RIGHT_EYE].mean(axis=0),
            lm[LM_NOSE_TIP],
            lm[LM_MOUTH_L],
            lm[LM_MOUTH_R],
        ], dtype=np.float32)
    return lm[:5].astype(np.float32)


def head_pose(lm, img_shape):
    """Yaw/pitch/roll in degrees via solvePnP. Returns (yaw, pitch, roll) or None."""
    if len(lm) < 468:
        return None
    h, w = img_shape[:2]
    pts2d = lm[_PNP_IDS].astype(np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts2d, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1])); yaw = np.degrees(np.arctan2(-R[2, 0], sy)); roll = 0.0
    # Normalise pitch/roll into [-180,180] then fold the 180-offset solvePnP often returns.
    pitch = ((pitch + 180) % 360) - 180
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return float(yaw), float(pitch), float(roll)


def align(img_bgr, lm, size=FACE_SIZE):
    """Similarity-transform the face to the canonical template.
    Returns (aligned image, aligned landmarks, affine matrix) or None."""
    src = five_anchors(lm)
    M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    warped = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    lm_h = np.hstack([lm, np.ones((len(lm), 1), dtype=np.float32)])
    lm_aligned = (lm_h @ M.T).astype(np.float32)
    return warped, lm_aligned, M


# ---------------------------------------------------------------------------
# 3. Region masks
# ---------------------------------------------------------------------------

def _hull_mask(points, size, dilate=0, blur=0):
    m = np.zeros((size, size), dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    if blur:
        k = blur | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def region_masks(lm_aligned, size=FACE_SIZE):
    """Semantic region masks in canonical space. Uses the median landmark layout so
    all frames share identical masks — that is what makes regions comparable."""
    if len(lm_aligned) < 468:
        # Fallback: coarse horizontal bands + vertical split.
        m = {}
        bands = {"forehead": (0.00, 0.28), "nose": (0.42, 0.66),
                 "mouth": (0.66, 0.84), "chin": (0.84, 1.00)}
        for name, (a, b) in bands.items():
            mask = np.zeros((size, size), np.uint8)
            mask[int(a * size):int(b * size), :] = 255
            m[name] = mask
        eye = np.zeros((size, size), np.uint8)
        eye[int(0.28 * size):int(0.42 * size), :] = 255
        m["left_eye"] = cv2.bitwise_and(eye, _half(size, True))
        m["right_eye"] = cv2.bitwise_and(eye, _half(size, False))
        m["left_cheek"] = np.zeros((size, size), np.uint8)
        m["right_cheek"] = np.zeros((size, size), np.uint8)
        return m, np.full((size, size), 255, np.uint8)

    face = _hull_mask(lm_aligned[LM_FACE_OVAL], size, dilate=5)
    masks = {
        "forehead": _hull_mask(lm_aligned[LM_FOREHEAD], size, dilate=7),
        "left_eye": _hull_mask(lm_aligned[LM_LEFT_EYE + LM_LEFT_BROW], size, dilate=9),
        "right_eye": _hull_mask(lm_aligned[LM_RIGHT_EYE + LM_RIGHT_BROW], size, dilate=9),
        "nose": _hull_mask(lm_aligned[LM_NOSE], size, dilate=7),
        "mouth": _hull_mask(lm_aligned[LM_MOUTH], size, dilate=11),
        "chin": _hull_mask(lm_aligned[LM_CHIN], size, dilate=7),
    }
    for k in masks:
        masks[k] = cv2.bitwise_and(masks[k], face)

    covered = np.zeros((size, size), np.uint8)
    for k in masks:
        covered = cv2.bitwise_or(covered, masks[k])
    leftover = cv2.bitwise_and(face, cv2.bitwise_not(covered))
    masks["left_cheek"] = cv2.bitwise_and(leftover, _half(size, True))
    masks["right_cheek"] = cv2.bitwise_and(leftover, _half(size, False))
    return masks, face


def _half(size, left):
    m = np.zeros((size, size), np.uint8)
    if left:
        m[:, : size // 2] = 255
    else:
        m[:, size // 2:] = 255
    return m


# ---------------------------------------------------------------------------
# 4. Reliability scoring
# ---------------------------------------------------------------------------

def _norm(v, lo, hi):
    return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def region_metrics(aligned_bgr, mask):
    """Raw per-region measurements. All from real pixels, no model involved."""
    idx = mask > 127
    if idx.sum() < 30:
        return None
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = gray[idx].astype(np.float64)
    mean = float(vals.mean())
    return {
        "sharpness": float(lap[idx].var()),
        "contrast": float(vals.std()),
        "mean_luma": mean,
        # penalise clipping: fraction of pixels stuck at 0 or 255
        "clipped": float(((vals <= 2) | (vals >= 253)).mean()),
    }


def score_region(met, geom_dev, pose_dist, sharp_ref, weights):
    """Weighted reliability in [0,1]. sharp_ref = 95th-pct sharpness across the set,
    so scoring is relative to the best evidence actually available."""
    if met is None:
        return 0.0
    s_sharp = _norm(met["sharpness"], 0.0, max(sharp_ref, 1e-6))
    s_contrast = _norm(met["contrast"], 4.0, 55.0)
    s_expose = 1.0 - _norm(abs(met["mean_luma"] - 128.0), 0.0, 110.0)
    s_clip = 1.0 - _norm(met["clipped"], 0.0, 0.25)
    s_geom = 1.0 - _norm(geom_dev, 1.0, 12.0)        # px RMS from the cluster median
    s_pose = 1.0 - _norm(pose_dist, 5.0, 45.0)       # degrees from the cluster centre
    w = weights
    total = (w["sharpness"] * s_sharp + w["contrast"] * s_contrast +
             w["exposure"] * s_expose + w["clipping"] * s_clip +
             w["geometry"] * s_geom + w["pose"] * s_pose)
    return float(np.clip(total / sum(w.values()), 0.0, 1.0))


DEFAULT_WEIGHTS = {"sharpness": 0.40, "contrast": 0.12, "exposure": 0.10,
                   "clipping": 0.08, "geometry": 0.18, "pose": 0.12}


# ---------------------------------------------------------------------------
# 5. Pose clustering
# ---------------------------------------------------------------------------

def dominant_pose_cluster(poses, tol_deg=20.0, frontal_bias=True):
    """Greedy clustering on (yaw,pitch). Returns (member indices, centre)."""
    valid = [i for i, p in enumerate(poses) if p is not None]
    if not valid:
        return list(range(len(poses))), (0.0, 0.0)
    arr = np.array([[poses[i][0], poses[i][1]] for i in valid], dtype=np.float64)
    best = None
    for k in range(len(valid)):
        d = np.linalg.norm(arr - arr[k], axis=1)
        members = d <= tol_deg
        centre = arr[members].mean(axis=0)
        # prefer the biggest cluster; break ties toward frontal
        key = (int(members.sum()), -float(np.linalg.norm(centre)) if frontal_bias else 0.0)
        if best is None or key > best[0]:
            best = (key, members, centre)
    _, members, centre = best
    idxs = [valid[i] for i in range(len(valid)) if members[i]]
    return idxs, (float(centre[0]), float(centre[1]))


# ---------------------------------------------------------------------------
# 6. Multiband (Laplacian pyramid) blending
# ---------------------------------------------------------------------------

def _pyr_down_list(img, levels):
    out = [img.astype(np.float32)]
    for _ in range(levels):
        out.append(cv2.pyrDown(out[-1]))
    return out


def _laplacian_pyr(img, levels):
    g = _pyr_down_list(img, levels)
    lap = []
    for i in range(levels):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def multiband_blend(images, weight_maps, levels=5):
    """images: list of HxWx3 float arrays. weight_maps: matching HxW float in [0,1],
    normalised across the list. Standard Burt-Adelson blending."""
    h, w = images[0].shape[:2]
    levels = min(levels, int(np.floor(np.log2(min(h, w)))) - 2)
    levels = max(levels, 1)
    stack = np.stack(weight_maps).astype(np.float32)
    stack /= np.clip(stack.sum(axis=0, keepdims=True), 1e-6, None)

    lap_pyrs = [_laplacian_pyr(im, levels) for im in images]
    w_pyrs = [_pyr_down_list(stack[i], levels) for i in range(len(images))]

    blended = []
    for lv in range(levels + 1):
        acc = np.zeros_like(lap_pyrs[0][lv])
        wsum = np.zeros(lap_pyrs[0][lv].shape[:2], dtype=np.float32)
        for i in range(len(images)):
            wm = cv2.resize(w_pyrs[i][lv], (lap_pyrs[i][lv].shape[1], lap_pyrs[i][lv].shape[0]))
            acc += lap_pyrs[i][lv] * wm[..., None]
            wsum += wm
        blended.append(acc / np.clip(wsum, 1e-6, None)[..., None])

    out = blended[-1]
    for lv in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[lv].shape[1], blended[lv].shape[0])) + blended[lv]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 7. Optional denoiser
# ---------------------------------------------------------------------------

def maybe_denoise(faces, checkpoint_path, batch_size=16, amp=True):
    """Clean each aligned crop with the trained restorer.

    Returns (faces, info). info is None when no checkpoint was supplied, otherwise a dict
    identifying exactly which model touched the evidence — checkpoint hash, architecture,
    training step, validation PSNR. A boolean "denoiser_applied: true" is not enough for a
    custody report: the defence is entitled to know *which* model, and to re-run it.

    Loads v4 checkpoints (NAFNet, EMA weights) through load_denoiser(), and falls back to
    the v3 loading path if train_denoiser_model.py predates it. Inference is batched and
    runs under autocast, which matters when a video gives you 200 crops rather than 6.
    """
    if not checkpoint_path:
        return faces, None
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, weights_used = None, {}, "model"

    try:
        from train_denoiser_model import load_denoiser          # v4 model file
        model = load_denoiser(checkpoint_path, device=device, prefer_ema=True)
        meta = dict(getattr(model, "meta", {}) or {})
        weights_used = "ema" if meta.get("arch") else "model"
        probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights_used = "ema" if isinstance(probe, dict) and "ema_state_dict" in probe \
            else "model"
        del probe
    except ImportError:
        from train_denoiser_model import FaceDenoiseNet         # v3 model file
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", "").replace("_orig_mod.", ""): v
                 for k, v in state.items()}
        kw = {"base": ckpt["base_width"]} if isinstance(ckpt, dict) and \
            "base_width" in ckpt else {}
        model = FaceDenoiseNet(**kw).to(device)
        model.load_state_dict(state)
        model.eval()
        meta = {k: v for k, v in (ckpt.items() if isinstance(ckpt, dict) else [])
                if not k.endswith("state_dict")}

    use_amp = amp and device.type == "cuda"
    dt = torch.float16
    if use_amp and torch.cuda.get_device_capability()[0] >= 8:
        dt = torch.bfloat16

    out, t0 = [], time.time()
    with torch.no_grad():
        for s in range(0, len(faces), batch_size):
            chunk = faces[s:s + batch_size]
            arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk])
            t = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(255)
            t = t.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=dt, enabled=use_amp):
                y = model(t)
            y = y.float().clamp(0, 1).mul_(255).round_().byte()
            y = y.permute(0, 2, 3, 1).cpu().numpy()
            out.extend(cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in y)

    info = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "architecture": meta.get("arch", "unet_v3"),
        "base_width": meta.get("base_width"),
        "parameters": meta.get("params"),
        "trained_steps": meta.get("step"),
        "val_psnr_db": meta.get("best_psnr"),
        "degradation_profile": meta.get("deg_mode"),
        "forensic_mode": meta.get("forensic_mode"),
        "weights_used": weights_used,
        "crops_processed": len(out),
        "device": str(device),
        "seconds": round(time.time() - t0, 3),
        "note": ("Denoising removes noise already present in the pixels. It is not "
                 "generative and adds no detail that was not measured."),
    }
    print(f"Denoiser: {info['architecture']} on {len(out)} crops in {info['seconds']}s "
          f"({info['weights_used']} weights, sha {info['checkpoint_sha256'][:12]})")
    return out, info


# ---------------------------------------------------------------------------
# 8. Pipeline
# ---------------------------------------------------------------------------

def reconstruct(input_path, outdir, max_images=200, denoiser_checkpoint=None,
                pose_tol=20.0, weights=None, min_region_score=0.25, levels=5):
    os.makedirs(outdir, exist_ok=True)
    weights = weights or DEFAULT_WEIGHTS
    t0 = datetime.now(timezone.utc)

    frames, labels = load_inputs(input_path, max_images=max_images)
    if not frames:
        raise ValueError("No usable frames.")

    det = LandmarkDetector()
    aligned, aligned_lm, poses, kept_labels, blendshapes = [], [], [], [], []
    for img, label in zip(frames, labels):
        lm, bs = det(img)
        if lm is None:
            continue
        a = align(img, lm)
        if a is None:
            continue
        warped, lm_a, _ = a
        aligned.append(warped)
        aligned_lm.append(lm_a)
        poses.append(head_pose(lm, img.shape))
        blendshapes.append(bs)
        kept_labels.append(label)

    print(f"Detector: {det.backend} | faces aligned: {len(aligned)}/{len(frames)}")
    if not aligned:
        raise RuntimeError("No face detected in any input. Check the footage or crop manually.")

    aligned, denoiser_info = maybe_denoise(aligned, denoiser_checkpoint)

    # --- pose clustering -----------------------------------------------------
    cluster_idx, centre = dominant_pose_cluster(poses, tol_deg=pose_tol)
    excluded_pose = [kept_labels[i] for i in range(len(aligned)) if i not in set(cluster_idx)]
    print(f"Pose cluster: {len(cluster_idx)}/{len(aligned)} frames around "
          f"yaw={centre[0]:.1f}deg pitch={centre[1]:.1f}deg "
          f"({len(excluded_pose)} excluded as pose-incompatible)")

    # --- expression clustering (only when blendshapes are available) ---------
    excluded_expr = []
    have_bs = [i for i in cluster_idx if blendshapes[i] is not None]
    if len(have_bs) >= 4:
        B = np.stack([blendshapes[i] for i in have_bs])
        med = np.median(B, axis=0)
        dist = np.linalg.norm(B - med, axis=1)
        keep_thresh = float(np.percentile(dist, 80))
        surviving = [have_bs[k] for k in range(len(have_bs)) if dist[k] <= keep_thresh]
        if len(surviving) >= 3:
            excluded_expr = [kept_labels[i] for i in cluster_idx if i not in set(surviving)]
            cluster_idx = surviving
            print(f"Expression cluster: kept {len(cluster_idx)} frames "
                  f"({len(excluded_expr)} dropped as expression-incompatible)")
    else:
        print("Expression clustering skipped (backend provides no blendshapes, or too few "
              "frames in the pose cluster).")

    # --- masks from the median landmark layout of the cluster ----------------
    lm_stack = np.stack([aligned_lm[i] for i in cluster_idx])
    lm_median = np.median(lm_stack, axis=0)
    masks, face_mask = region_masks(lm_median)

    # --- scoring -------------------------------------------------------------
    sharp_samples = []
    for i in cluster_idx:
        g = cv2.cvtColor(aligned[i], cv2.COLOR_BGR2GRAY)
        sharp_samples.append(cv2.Laplacian(g, cv2.CV_64F)[face_mask > 127].var())
    sharp_ref = float(np.percentile(sharp_samples, 95)) if sharp_samples else 1.0

    geom_dev = {i: float(np.sqrt(((aligned_lm[i] - lm_median) ** 2).sum(axis=1).mean()))
                for i in cluster_idx}
    pose_dist = {}
    for i in cluster_idx:
        p = poses[i]
        pose_dist[i] = 0.0 if p is None else float(np.hypot(p[0] - centre[0], p[1] - centre[1]))

    scores = {r: {} for r in REGION_ORDER}
    for r in REGION_ORDER:
        if masks[r].max() == 0:
            continue
        for i in cluster_idx:
            met = region_metrics(aligned[i], masks[r])
            scores[r][i] = score_region(met, geom_dev[i], pose_dist[i], sharp_ref, weights)

    # --- base frame: best average across all regions -------------------------
    per_frame_mean = {i: float(np.mean([scores[r].get(i, 0.0) for r in REGION_ORDER
                                        if scores[r]])) for i in cluster_idx}
    base_idx = max(per_frame_mean, key=per_frame_mean.get)

    # --- select best frame per region ----------------------------------------
    selection, confidence = {}, {}
    for r in REGION_ORDER:
        if not scores[r]:
            selection[r], confidence[r] = base_idx, 0.0
            continue
        best_i = max(scores[r], key=scores[r].get)
        selection[r] = best_i
        confidence[r] = scores[r][best_i]

    # --- build weight maps and blend -----------------------------------------
    used = sorted(set(selection.values()) | {base_idx})
    wmaps = {i: np.zeros((FACE_SIZE, FACE_SIZE), np.float32) for i in used}
    wmaps[base_idx] += 0.20  # gentle base everywhere, so nothing is ever empty
    attribution_img = np.zeros((FACE_SIZE, FACE_SIZE, 3), np.uint8)
    low_conf_regions = []

    for r in REGION_ORDER:
        m = masks[r]
        if m.max() == 0:
            continue
        soft = cv2.GaussianBlur(m, (21, 21), 0).astype(np.float32) / 255.0
        if confidence[r] < min_region_score:
            low_conf_regions.append(r)
        wmaps[selection[r]] += soft
        attribution_img[m > 127] = REGION_COLORS[r]

    composite = multiband_blend([aligned[i].astype(np.float32) for i in used],
                                [wmaps[i] for i in used], levels=levels)

    # --- outputs -------------------------------------------------------------
    comp_path = os.path.join(outdir, "composite.png")
    cv2.imwrite(comp_path, composite)

    overlay = cv2.addWeighted(composite, 0.55, attribution_img, 0.45, 0)
    for r in REGION_ORDER:
        if masks[r].max() == 0:
            continue
        ys, xs = np.where(masks[r] > 127)
        cv2.putText(overlay, f"{r}:{kept_labels[selection[r]][:10]}",
                    (max(2, int(xs.mean()) - 40), int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    overlay_path = os.path.join(outdir, "attribution_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    contact = np.hstack([cv2.resize(aligned[i], (128, 128)) for i in used[:8]])
    cv2.imwrite(os.path.join(outdir, "sources_used.png"), contact)

    report = {
        "generated_utc": t0.isoformat(),
        "tool": "face_fusion_v2",
        "detector_backend": det.backend,
        "input": os.path.abspath(input_path),
        "input_sha256": sha256_of(input_path) if os.path.isfile(input_path) else None,
        "frames_supplied": len(frames),
        "faces_detected": len(aligned),
        "pose_cluster_size": len(cluster_idx),
        "pose_cluster_centre_deg": {"yaw": centre[0], "pitch": centre[1]},
        "pose_tolerance_deg": pose_tol,
        "excluded_pose_incompatible": excluded_pose,
        "excluded_expression_incompatible": excluded_expr,
        "fusion_cluster_size": len(cluster_idx),
        "denoiser_applied": bool(denoiser_info),
        "denoiser": denoiser_info,
        "base_frame": kept_labels[base_idx],
        "weights": weights,
        "attribution": {
            r: {
                "source": kept_labels[selection[r]],
                "source_index": int(selection[r]),
                "confidence": round(confidence[r], 4),
                "flag": ("low_confidence" if confidence[r] < min_region_score else "ok"),
                "runner_up": (sorted(scores[r].items(), key=lambda kv: -kv[1])[1:2] and
                              {"source": kept_labels[sorted(scores[r].items(),
                                                            key=lambda kv: -kv[1])[1][0]],
                               "confidence": round(sorted(scores[r].items(),
                                                          key=lambda kv: -kv[1])[1][1], 4)}) or None,
            } for r in REGION_ORDER if scores[r]
        },
        "low_confidence_regions": low_conf_regions,
        "no_hallucination": ("Every output pixel is a Laplacian-pyramid blend of pixels from "
                             "the listed source frames. No generative model was used."),
        "outputs": {},
    }
    report["outputs"] = {
        "composite.png": sha256_of(comp_path),
        "attribution_overlay.png": sha256_of(overlay_path),
    }
    report_path = os.path.join(outdir, "attribution.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nComposite:   {comp_path}")
    print(f"Overlay:     {overlay_path}")
    print(f"Report:      {report_path}")
    for r in REGION_ORDER:
        if r in report["attribution"]:
            a = report["attribution"][r]
            print(f"  {r:<12} <- {a['source']:<20} conf {a['confidence']:.2f} {a['flag']}")
    if low_conf_regions:
        print(f"\n[!] Low-confidence regions (treat as unreliable evidence): "
              f"{', '.join(low_conf_regions)}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Video file OR folder of images")
    ap.add_argument("--outdir", default="fusion_out")
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--pose_tol", type=float, default=20.0,
                    help="Degrees of yaw/pitch spread allowed inside one fusion cluster")
    ap.add_argument("--min_region_score", type=float, default=0.25)
    ap.add_argument("--denoiser_checkpoint", default=None)
    a = ap.parse_args()
    reconstruct(a.input, a.outdir, a.max_images, a.denoiser_checkpoint,
                a.pose_tol, None, a.min_region_score)
