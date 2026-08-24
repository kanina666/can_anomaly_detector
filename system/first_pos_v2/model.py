"""
first_pos v2 anomaly signal: whole-image ResNet18 OR (row12 / row0 zone
PatchCore-AND-HSV signal). Developed 2026-08-20 to replace the per-cell
ensemble for first_pos only -- second_pos is untouched (see top-level
can_anomaly_detector.check_anomaly).

On the two honest held-out test sets (never touched by any training or
threshold-calibration step here):
  audit_240   (120 norma + 120 anomaly, the official first_pos audit set):
      recall=100.0% (120/120)  specificity=95.8% (115/120, fp=5)
  newdate_281 (241 norma + 40 anomaly, 2026-07-31, a camera-reposition date
               never seen by any component below):
      recall=100.0% (40/40)    specificity=97.1% (234/241, fp=7)
  pooled (n=521): recall=100.0% specificity=96.7% precision=93.0% accuracy=97.7%
vs. the per-cell ensemble it replaces, on the same newdate_281 photos:
  recall=47.5% specificity=86.3% (calibration/mask did not survive the
  camera reposition -- see CLAUDE.md, "first_pos: сессия 2026-08-20").

ARCHITECTURE:
  is_anomaly = (whole_image_resnet_prob >= whole_image_threshold)
               OR row12_signal OR row0_signal

  row12_signal / row0_signal = True iff there exists at least one column in
  that row where: resnet_patchcore_score >= thr AND dino_patchcore_score >=
  thr AND the patch does not look like a glare/highlight (HSV-thresholded).

Known remaining gap (not fixed -- do not re-attempt without new labeled
data, see CLAUDE.md "first_pos: сессия 2026-08-20" п.14-23 for the seven
approaches already tried and why they failed): ~9/521 row12 false positives
on edge/corner columns (0, 11-15) -- score distributions genuinely overlap
with real edge-column gaps on every axis tried.

NOT VALIDATED on real Raspberry Pi 5 hardware -- DINOv2 ViT-S/14 is much
heavier than anything else in this codebase. Benchmark on target hardware
before relying on the 3s production limit.

Model weights (dinov2_vits14_pretrain.pth, resnet18_imagenet.pth) and the
four PatchCore memory banks (~945MB total) are NOT committed to git -- they
live in artifacts_external/, fetched by download_artifacts.py. See
system/README.md.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .dinov2.models.vision_transformer import vit_small

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(HERE)
ART = os.path.join(HERE, "artifacts")
ART_EXT = os.path.join(HERE, "artifacts_external")

with open(os.path.join(SYSTEM_DIR, "config.json"), encoding="utf-8") as f:
    _CFG = json.load(f)["first_pos_v2"]

GRID = _CFG["grid"]
INPUT_SIZE_PC = _CFG["input_size_patchcore"]
INPUT_SIZE_WHOLE = _CFG["input_size_whole_image"]
HALF = {int(k): v for k, v in _CFG["half_size_by_row"].items()}
SPEC_TARGET = _CFG["spec_target"]
WHOLE_IMAGE_THRESHOLD = _CFG["whole_image_threshold"]
ROWS = tuple(HALF.keys())

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_state = {}


def _missing_external_artifacts():
    required = ["dinov2_vits14_pretrain.pth", "resnet18_imagenet.pth"] + [
        f"row{row}_{kind}_bank.npy" for row in ROWS for kind in ("resnet", "dino")
    ]
    return [name for name in required if not os.path.exists(os.path.join(ART_EXT, name))]


def _build_resnet_whole_image(device):
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    ckpt = torch.load(os.path.join(ART, "resnet18_whole_image.pth"), map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval().to(device)
    return m


def _build_resnet_patchcore_extractor(device):
    m = tvm.resnet18(weights=None)
    state_dict = torch.load(os.path.join(ART_EXT, "resnet18_imagenet.pth"), map_location=device)
    m.load_state_dict(state_dict)
    m.eval().to(device)
    feats = {}

    def hook(name):
        def fn(_, __, out):
            feats[name] = out.detach()
        return fn
    m.layer2.register_forward_hook(hook("l2"))
    m.layer3.register_forward_hook(hook("l3"))

    @torch.no_grad()
    def extract(t):
        feats.clear()
        m(t)
        v2 = F.adaptive_avg_pool2d(feats["l2"], GRID)
        v3 = F.adaptive_avg_pool2d(feats["l3"], GRID)
        v = torch.cat([v2, v3], dim=1)
        B, C, H, W = v.shape
        v = v.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return F.normalize(v, dim=2)
    return extract


def _build_dino_extractor(device):
    m = vit_small(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
                   block_chunks=0, num_register_tokens=0,
                   interpolate_antialias=False, interpolate_offset=0.1)
    state_dict = torch.load(os.path.join(ART_EXT, "dinov2_vits14_pretrain.pth"), map_location=device)
    m.load_state_dict(state_dict, strict=True)
    m.eval().to(device)
    n_side = INPUT_SIZE_PC // 14

    @torch.no_grad()
    def extract(t):
        out = m.forward_features(t)
        tokens = out["x_norm_patchtokens"]
        B, N, D = tokens.shape
        grid = tokens.reshape(B, n_side, n_side, D).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid, GRID)
        v = pooled.permute(0, 2, 3, 1).reshape(B, GRID * GRID, D)
        return F.normalize(v, dim=2)
    return extract


def _load(device):
    if device in _state:
        return _state[device]

    missing = _missing_external_artifacts()
    if missing:
        raise FileNotFoundError(
            "first_pos_v2: missing external artifacts (not committed to git): "
            f"{missing}. Run: python first_pos_v2/download_artifacts.py "
            "(see system/README.md)."
        )

    mean = MEAN.to(device)
    std = STD.to(device)

    whole_model = _build_resnet_whole_image(device)
    resnet_ex = _build_resnet_patchcore_extractor(device)
    dino_ex = _build_dino_extractor(device)

    banks = {}
    thr_r, thr_d = {}, {}
    for row in ROWS:
        banks[(row, "resnet")] = torch.from_numpy(
            np.load(os.path.join(ART_EXT, f"row{row}_resnet_bank.npy"))).to(device)
        banks[(row, "dino")] = torch.from_numpy(
            np.load(os.path.join(ART_EXT, f"row{row}_dino_bank.npy"))).to(device)
        thr_r[row] = json.load(open(os.path.join(ART, f"row{row}_resnet_meta.json"),
                                     encoding="utf-8"))["thr_by_spec"][SPEC_TARGET]
        thr_d[row] = json.load(open(os.path.join(ART, f"row{row}_dino_meta.json"),
                                     encoding="utf-8"))["thr_by_spec"][SPEC_TARGET]

    hsv_thr = json.load(open(os.path.join(ART, "hsv_safe_thresholds.json"), encoding="utf-8"))

    calib = json.load(open(os.path.join(SYSTEM_DIR, "calib_yolo_first_pos.json"), encoding="utf-8"))["cells"]
    row_cols = {row: {c["col"]: (c["x"], c["y"]) for c in calib if c["row"] == row} for row in ROWS}

    _state[device] = dict(
        whole_model=whole_model, resnet_ex=resnet_ex, dino_ex=dino_ex,
        banks=banks, thr_r=thr_r, thr_d=thr_d, hsv_thr=hsv_thr, row_cols=row_cols,
        mean=mean, std=std,
    )
    return _state[device]


def _get_patch(img, x, y, half):
    h, w = img.shape[:2]
    x0, y0 = int(round(x - half)), int(round(y - half))
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch = img[y0c:y1c, x0c:x1c]
    if patch.shape[0] < 2 * half or patch.shape[1] < 2 * half:
        canvas = np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)
        oy, ox = y0c - y0, x0c - x0
        canvas[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
        patch = canvas
    return patch


def _patch_to_array(patch, size):
    p = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(p, (2, 0, 1))


def _hsv_stats(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    glare_mask = (v > 0.85) & (s < 0.25)
    return {"mean_s": float(s.mean()), "mean_v": float(v.mean()), "glare_frac": float(glare_mask.mean())}


def _looks_like_glare(st, hsv_thr):
    return (st["mean_v"] > hsv_thr["thr_v"]) or (st["glare_frac"] > hsv_thr["thr_g"]) or (st["mean_s"] < hsv_thr["thr_s"])


def _score(v, bank):
    flat = v.reshape(-1, v.shape[-1])
    max_sim = (flat @ bank.T).max(dim=1).values
    return (1.0 - max_sim).reshape(v.shape[0], v.shape[1]).max(dim=1).values.cpu().numpy()


def _row_signal(img, row, state):
    """True iff at least one column in this row is flagged by BOTH PatchCore
    feature spaces AND does not look like a glare artifact.

    Lazy DINOv2 evaluation: DINOv2 is ~4x more expensive than the ResNet
    extractor, and the AND condition means a column can only ever trigger
    if ResNet already cleared its own threshold there. So ResNet-PatchCore
    runs on all 16 columns first (cheap); DINOv2 only runs on the subset
    that already passed the ResNet gate (empty on a typical clean photo).
    """
    row_cols = state["row_cols"][row]
    device = next(state["whole_model"].parameters()).device
    cols = sorted(row_cols.keys())

    batch = [_patch_to_array(_get_patch(img, row_cols[c][0], row_cols[c][1], HALF[row]), INPUT_SIZE_PC)
             for c in cols]
    t = torch.from_numpy(np.stack(batch)).to(device)
    t_n = (t - state["mean"]) / state["std"]

    with torch.no_grad():
        v_r = state["resnet_ex"](t_n)
    scores_r = _score(v_r, state["banks"][(row, "resnet")])
    resnet_pass = [i for i, c in enumerate(cols) if scores_r[i] >= state["thr_r"][row]]
    if not resnet_pass:
        return False, [], {}

    with torch.no_grad():
        v_d_subset = state["dino_ex"](t_n[resnet_pass])
    scores_d_subset = _score(v_d_subset, state["banks"][(row, "dino")])

    triggering = [cols[resnet_pass[j]] for j, i in enumerate(resnet_pass)
                  if scores_d_subset[j] >= state["thr_d"][row]]
    if not triggering:
        return False, [], {}

    survivors = []
    scores = {}
    for c in triggering:
        cx, cy = row_cols[c]
        patch = _get_patch(img, cx, cy, HALF[row])
        st = _hsv_stats(patch)
        if not _looks_like_glare(st, state["hsv_thr"]):
            survivors.append(c)
            scores[c] = {"x": cx, "y": cy}
    return len(survivors) > 0, survivors, scores


def check_anomaly(img, device="cpu"):
    """img: BGR ndarray (already loaded via cv2.imread by the caller).
    Returns is_anomaly, whole_image_probability, and flagged row12/row0
    columns with their (x, y) calibrated location.
    """
    state = _load(device)
    torch_device = next(state["whole_model"].parameters()).device

    whole_arr = _patch_to_array(img, INPUT_SIZE_WHOLE)
    t = torch.from_numpy(whole_arr).unsqueeze(0).to(torch_device)
    t = (t - state["mean"]) / state["std"]
    with torch.no_grad():
        logit = state["whole_model"](t).item()
    whole_prob = 1.0 / (1.0 + np.exp(-logit))

    row_flags = {}
    for row in ROWS:
        flag, cols, locations = _row_signal(img, row, state)
        row_flags[row] = {"flag": flag, "cols": cols, "locations": locations}

    is_anomaly = bool(whole_prob >= WHOLE_IMAGE_THRESHOLD or any(row_flags[r]["flag"] for r in ROWS))

    return {
        "is_anomaly": is_anomaly,
        "whole_image_probability": round(float(whole_prob), 4),
        "row_flags": row_flags,
    }


def warm(device="cpu"):
    """Force model/bank load AND one full dummy forward pass through every
    code path (whole-image, both PatchCore extractors for both rows) ahead
    of the first real request.

    Measured on a dev laptop: even after _load() alone, the first real
    check_anomaly() call is ~2.5s slower than steady state (~0.6s clean /
    ~1.6s triggered) -- PyTorch's own first-call kernel/allocator warmup,
    separate from weight loading. That extra 2.5s on a clean photo alone
    would already exceed the 3s production budget if it landed on a real
    request. Running a dummy forward here (during --service startup, not
    per-request) absorbs that cost once instead.
    """
    state = _load(device)
    dummy = np.zeros((INPUT_SIZE_WHOLE, INPUT_SIZE_WHOLE, 3), dtype=np.uint8)

    t = torch.from_numpy(_patch_to_array(dummy, INPUT_SIZE_WHOLE)).unsqueeze(0).to(device)
    t = (t - state["mean"]) / state["std"]
    with torch.no_grad():
        state["whole_model"](t)

    for row in ROWS:
        n = len(state["row_cols"][row])
        patch_dummy = np.zeros((INPUT_SIZE_PC, INPUT_SIZE_PC, 3), dtype=np.uint8)
        batch = np.stack([_patch_to_array(patch_dummy, INPUT_SIZE_PC)] * n)
        t = torch.from_numpy(batch).to(device)
        t_n = (t - state["mean"]) / state["std"]
        with torch.no_grad():
            v_r = state["resnet_ex"](t_n)
            _score(v_r, state["banks"][(row, "resnet")])
            v_d = state["dino_ex"](t_n)
            _score(v_d, state["banks"][(row, "dino")])
