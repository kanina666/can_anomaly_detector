import json
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbox_config import in_matrix

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

ZONE_MID = CONFIG["zone_mid"]
HALF_SIZE = CONFIG["half_size"]
YOLO_ZONE_CONF = CONFIG["yolo_zone_conf"]
EXPECTED_COUNT = CONFIG["expected_count"]
INPUT_SIZE = CONFIG["percell_input_size"]
CHRONIC_CELLS = {pos: set(tuple(rc) for rc in CONFIG["reliability_mask_chronic_cells"][pos])
                  for pos in ("first_pos", "second_pos")}
LOCATION_CONFIDENCE_THRESHOLD = CONFIG.get("location_confidence_threshold", 0.99)
ENSEMBLE_TYPE = CONFIG.get("ensemble_type", {"first_pos": "logreg", "second_pos": "logreg"})

_yolo_model = None
_percell_models = {}
_calib_cache = {}
_lgbm_models = {}


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)


def _normalize_device(device):
    d = str(device).lower()
    if d == "cpu":
        return "cpu", "cpu"
    if d == "cuda":
        return "cuda", "0"
    if d.startswith("cuda:"):
        return d, d.split(":", 1)[1]
    return f"cuda:{d}", d


def _get_yolo(device):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG["yolo_model_path"])
        _yolo_model = YOLO(model_path)
    return _yolo_model


def _get_percell(device):
    torch_device, _ = _normalize_device(device)
    if torch_device not in _percell_models:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG["percell_model_path"])
        ckpt = torch.load(model_path, map_location=torch_device)
        m = TinyCNN().to(torch_device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _percell_models[torch_device] = m
    return _percell_models[torch_device]


def _get_calib(position):
    if position not in _calib_cache:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"calib_yolo_{position}.json")
        with open(path, encoding="utf-8") as f:
            _calib_cache[position] = json.load(f)
    return _calib_cache[position]


def _get_lgbm(position):
    if position not in _lgbm_models:
        cfg = CONFIG["ensemble_lgbm"][position]
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["model_file"])
        _lgbm_models[position] = lgb.Booster(model_file=model_path)
    return _lgbm_models[position]


def _row_to_zone3(row):
    if row <= 3:
        return "near"
    if row <= 8:
        return "mid"
    return "far"


def _zone_of(y, position):
    return "far" if y < ZONE_MID[position] else "near"


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


def _yolo_count(image_path, folder, position, yolo_device):
    model = _get_yolo(yolo_device)
    r = model.predict(image_path, imgsz=1280, conf=0.05, iou=0.7, max_det=3000,
                       device=yolo_device, verbose=False)[0]
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cnt = 0
    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        thr = YOLO_ZONE_CONF["far"] if _zone_of(cy, position) == "far" else YOLO_ZONE_CONF["near"]
        if c >= thr and in_matrix(cx, cy, folder):
            cnt += 1
    return cnt


def _percell_probs(img, position, torch_device):
    model = _get_percell(torch_device)
    calib = _get_calib(position)
    cells = calib["cells"]
    batch = []
    for c in cells:
        half = HALF_SIZE[position][_zone_of(c["y"], position)]
        patch = _get_patch(img, c["x"], c["y"], half)
        patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(np.transpose(patch, (2, 0, 1)))
    t = torch.from_numpy(np.stack(batch)).to(torch_device)
    with torch.no_grad():
        logits = model(t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(logits))
    return cells, probs


def check_anomaly(image_path, position, device="cpu"):
    if position not in ("first_pos", "second_pos"):
        raise ValueError(f"position must be 'first_pos' or 'second_pos', got: {position}")

    folder = f"{position}_anomaly"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    torch_device, yolo_device = _normalize_device(device)
    detected_count = _yolo_count(image_path, folder, position, yolo_device)
    expected_count = EXPECTED_COUNT[position]
    yolo_deficit = expected_count - detected_count
    cells, probs = _percell_probs(img, position, torch_device)

    chronic = CHRONIC_CELLS[position]
    reliable_idx = [i for i, c in enumerate(cells) if (c["row"], c["col"]) not in chronic]
    reliable_probs = probs[reliable_idx]
    top3 = np.sort(reliable_probs)[-3:]
    max_p = float(reliable_probs.max())
    sum_top3 = float(top3.sum())

    if ENSEMBLE_TYPE[position] == "lightgbm":
        row12_idx = [i for i, c in enumerate(cells) if c["row"] == 12]
        p_row12 = probs[row12_idx]
        feat_names = CONFIG["ensemble_lgbm"][position]["feature_names"]
        feat_map = {
            "yolo_deficit": float(yolo_deficit),
            "max_masked": max_p,
            "sum_top3_masked": sum_top3,
            "max_row12": float(p_row12.max()) if len(p_row12) else 0.0,
            "mean_row12": float(p_row12.mean()) if len(p_row12) else 0.0,
            "count_above_0.3_masked": int((reliable_probs > 0.3).sum()),
            "count_above_0.5_masked": int((reliable_probs > 0.5).sum()),
            "count_above_0.7_masked": int((reliable_probs > 0.7).sum()),
        }
        X = np.array([[feat_map[f] for f in feat_names]], dtype=float)
        booster = _get_lgbm(position)
        probability = float(booster.predict(X)[0])
    else:
        logreg = CONFIG["ensemble_logreg"][position]
        feat = np.array([yolo_deficit, max_p, sum_top3], dtype=float)
        mean = np.array(logreg["feature_mean"])
        std = np.array(logreg["feature_std"])
        feat_s = (feat - mean) / std
        w = np.array(logreg["weights"])
        z = w[0] + np.dot(w[1:], feat_s)
        probability = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    threshold = CONFIG["decision_threshold"][position]["threshold"]
    is_anomaly = bool(probability > threshold)

    order = np.argsort(reliable_probs)[::-1][:3]
    top3_locations = []
    for oi in order:
        ci = reliable_idx[oi]
        c = cells[ci]
        top3_locations.append({"x": round(c["x"], 1), "y": round(c["y"], 1),
                                "row": c["row"], "col": c["col"],
                                "prob_missing": round(float(probs[ci]), 4)})

    location_confidence = "high" if max_p >= LOCATION_CONFIDENCE_THRESHOLD else "low"

    result = {
        "is_anomaly": is_anomaly,
        "probability": round(float(probability), 4),
        "threshold": threshold,
        "detected_count": int(detected_count),
        "expected_count": int(expected_count),
        "location_confidence": location_confidence,
        "top3_candidate_locations": top3_locations,
    }

    if position == "second_pos" and location_confidence == "low":
        top1_row = top3_locations[0]["row"]
        result["zone"] = {
            "value": _row_to_zone3(top1_row),
            "note": "Low confidence in the exact cell - the general zone is a more reliable guide here.",
        }

    return result


def run_as_service():
    device = "cpu"
    _get_yolo(device)
    _get_percell(device)
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            print(json.dumps({"error": f"bad input line: {line!r}"}), flush=True)
            continue
        image_path, position = parts
        try:
            result = check_anomaly(image_path, position, device=device)
        except Exception as e:
            result = {"error": str(e)}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--service":
        run_as_service()
    elif len(sys.argv) == 3:
        result = check_anomaly(sys.argv[1], sys.argv[2])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Usage:")
        print("  python can_anomaly_detector.py <photo.jpg> <first_pos|second_pos>")
        print("  python can_anomaly_detector.py --service   (reads stdin lines: 'path.jpg position')")
        sys.exit(1)
