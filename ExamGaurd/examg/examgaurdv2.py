"""ExamGuard (Lite) — optimized detection loop + Feature #1 incident logic

Focus of this version:
- Reduce CPU/GPU overhead in the per-frame loop.
- Remove all clip/export/FFmpeg segment complexity for now.
- Keep the important proctoring logic (Feature #1):
  - Dwell confirmation (debounce)
  - Cooldown (+ overrides)
  - Severity tiers (warning / incident / critical)

Performance knobs (in config):
- yolo.imgsz (smaller = faster)
- yolo.infer_stride (run model every N frames; reuse last result in-between)
- yolo.use_fp16 (CUDA only)
- yolo.max_det (limit detections)

Run:
  python examguard.py
  python examguard.py --config examguard_config.json

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==============================
# Config
# ==============================

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "video_source": 1,
        "width": 2514,
        "height": 1080,
        "backend": "CAP_DSHOW",
        "buffer_size": 1,
    },
    "yolo": {
        "model_path": "best.pt",
        "tracker_yaml": "bytetrack_custom.yaml",
        "imgsz": 640,
        "conf": 0.20,
        "infer_stride": 1,
        "max_det": 10,
        "use_fp16": True,
    },
    "output": {
        "report_dir": "exam_reports",
        "write_csv": True,
    },
    "logic": {
        "risk_levels": {
            "normal": -2,
            "looking_left": 3,
            "looking_right": 3,
            "looking_back": 5,
            "phone": 10,
            "mobile": 10,
        },
        "warn_thresh": 60,
        "incident_thresh": 75,
        "critical_thresh": 90,
        "stop_hysteresis": 20,
        "consecutive_normal_to_stop": 45,
        "dwell_frames": {
            "phone": 3,
            "looking_left": 6,
            "looking_right": 6,
            "looking_back": 8,
        },
        "default_dwell": 6,
        "cooldown_seconds": 3.0,
        "cooldown_overrides": ["phone"],
        "max_incident_seconds": 30.0,
        "lost_timeout_frames": 30,
    },
    "ui": {
        "window_name": "ExamGuard Dashboard",
        "show_window": True,
        "font_scale": 0.6,
        "line_thickness": 2,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _coerce_video_source(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        return s
    return v


def _backend_from_name(name: Optional[str]) -> int:
    if not name:
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    name = str(name).strip().upper()
    return {
        "CAP_ANY": cv2.CAP_ANY,
        "CAP_DSHOW": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "CAP_MSMF": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "CAP_V4L2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "CAP_AVFOUNDATION": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
    }.get(name, cv2.CAP_ANY)


def load_config(path: str, write_default_if_missing: bool = True) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        if not write_default_if_missing:
            raise FileNotFoundError(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[INFO] Config not found. Wrote default config to: {path}")
        return json.loads(json.dumps(DEFAULT_CONFIG))

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    if isinstance(user_cfg, dict):
        merged = deep_update(merged, user_cfg)

    merged["camera"]["video_source"] = _coerce_video_source(merged["camera"].get("video_source"))
    return merged


# ==============================
# Logic
# ==============================


@dataclass
class LogicParams:
    risk_levels: Dict[str, int]
    warn_thresh: int
    incident_thresh: int
    critical_thresh: int
    stop_hysteresis: int
    consecutive_normal_to_stop: int
    dwell_frames: Dict[str, int]
    default_dwell: int
    cooldown_seconds: float
    cooldown_overrides: set[str]
    max_incident_seconds: float
    lost_timeout_frames: int


class EventLogger:
    def __init__(self, report_dir: str, write_csv: bool = True):
        self.write_csv = bool(write_csv)
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_path = os.path.join(report_dir, self.session_id)
        os.makedirs(self.base_path, exist_ok=True)

        self.csv_path = os.path.join(self.base_path, "event_log.csv")
        if self.write_csv:
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("Type,StartISO,EndISO,ID,Behavior,PeakSuspicion\n")

        print(f"[INFO] Session started. Reports: {self.base_path}")

    def log(self, event_type: str, sid: int, behavior: str, start_ts: float, end_ts: float, peak: int) -> None:
        if not self.write_csv:
            return
        start_iso = datetime.fromtimestamp(start_ts).isoformat()
        end_iso = datetime.fromtimestamp(end_ts).isoformat()
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{event_type},{start_iso},{end_iso},{sid},{behavior},{peak}\n")


class StudentMonitor:
    def __init__(self, sid: int, logic: LogicParams):
        self.id = sid
        self.logic = logic

        self.suspicion = 0
        self.behavior = "normal"
        self.color = (0, 255, 0)

        self.last_seen_frame = 0
        self.consecutive_normal = 0

        # Dwell (debounce)
        self._raw_streak_behavior = "normal"
        self._raw_streak_len = 0

        # Incident state
        self.incident_active = False
        self.incident_active_since_ts: Optional[float] = None
        self.incident_peak = 0
        self.incident_peak_behavior = "normal"

        self.cooldown_until_ts = 0.0
        self.warning_armed = True
        self.critical_armed = True

    def _confirm(self, raw: str) -> str:
        if raw == self._raw_streak_behavior:
            self._raw_streak_len += 1
        else:
            self._raw_streak_behavior = raw
            self._raw_streak_len = 1

        if raw == "normal":
            return "normal"

        req = self.logic.dwell_frames.get(raw, self.logic.default_dwell)
        return raw if self._raw_streak_len >= req else "normal"

    def _apply_score(self, confirmed: str) -> None:
        self.behavior = confirmed

        if confirmed == "normal":
            self.consecutive_normal += 1
        else:
            self.consecutive_normal = 0

        step = self.logic.risk_levels.get(confirmed, 0)
        self.suspicion = max(0, min(100, self.suspicion + step))

        # UI color (cheap)
        if self.suspicion < 30:
            self.color = (0, 255, 0)
        elif self.suspicion < self.logic.incident_thresh:
            self.color = (0, 165, 255)
        else:
            self.color = (0, 0, 255)

    def update(self, now_ts: float, raw_behavior: str) -> Tuple[str, Optional[str]]:
        """Update state for this frame.

        Returns: (confirmed_behavior, event)
        where event is one of: None, "WARNING", "INCIDENT_START", "INCIDENT_END", "CRITICAL"
        """
        confirmed = self._confirm(raw_behavior)
        self._apply_score(confirmed)

        # Peak during incident (for logging)
        if self.suspicion > self.incident_peak:
            self.incident_peak = self.suspicion
            self.incident_peak_behavior = confirmed

        # WARNING (edge-trigger)
        if self.suspicion < (self.logic.warn_thresh - 10):
            self.warning_armed = True
        if self.warning_armed and self.suspicion >= self.logic.warn_thresh and confirmed != "normal":
            self.warning_armed = False
            return confirmed, "WARNING"

        # CRITICAL (edge-trigger)
        if self.suspicion < (self.logic.critical_thresh - 10):
            self.critical_armed = True
        if self.critical_armed and self.suspicion >= self.logic.critical_thresh and confirmed != "normal":
            self.critical_armed = False
            return confirmed, "CRITICAL"

        # Incident start
        in_cooldown = now_ts < self.cooldown_until_ts
        override = confirmed in self.logic.cooldown_overrides

        if (
            self.suspicion >= self.logic.incident_thresh
            and (not self.incident_active)
            and confirmed != "normal"
            and ((not in_cooldown) or override)
        ):
            self.incident_active = True
            self.incident_active_since_ts = now_ts
            self.incident_peak = max(self.incident_peak, self.suspicion)
            self.incident_peak_behavior = confirmed
            return confirmed, "INCIDENT_START"

        # Incident end
        if self.incident_active:
            should_end = False
            if self.suspicion < (self.logic.incident_thresh - self.logic.stop_hysteresis):
                should_end = True
            elif self.consecutive_normal > self.logic.consecutive_normal_to_stop:
                should_end = True
            if self.incident_active_since_ts is not None and (now_ts - self.incident_active_since_ts) > self.logic.max_incident_seconds:
                should_end = True

            if should_end:
                self.incident_active = False
                self.cooldown_until_ts = now_ts + self.logic.cooldown_seconds
                return confirmed, "INCIDENT_END"

        return confirmed, None


# ==============================
# UI
# ==============================


def draw_violation(frame: np.ndarray, box_xyxy: np.ndarray, text: str) -> None:
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(frame, (x1, y1 - 45), (x1 + w + 20, y1), (0, 0, 255), -1)
    cv2.putText(frame, text, (x1 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


# ==============================
# Main
# ==============================


def normalize_behavior(name: str) -> str:
    raw = name.strip().lower().replace(" ", "_")
    if "phone" in raw or "mobile" in raw:
        return "phone"
    return raw


def setup_torch_perf() -> None:
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def main(config_path: str, write_default_config: bool) -> None:
    cfg = load_config(config_path, write_default_if_missing=write_default_config)

    report_dir = str(cfg_get(cfg, "output.report_dir", "exam_reports"))
    write_csv = bool(cfg_get(cfg, "output.write_csv", True))
    logger = EventLogger(report_dir=report_dir, write_csv=write_csv)

    cam_source = _coerce_video_source(cfg_get(cfg, "camera.video_source", 1))
    cam_w = int(cfg_get(cfg, "camera.width", 1280))
    cam_h = int(cfg_get(cfg, "camera.height", 720))
    backend = _backend_from_name(cfg_get(cfg, "camera.backend", "CAP_DSHOW"))
    buffer_size = int(cfg_get(cfg, "camera.buffer_size", 1))

    model_path = str(cfg_get(cfg, "yolo.model_path", "best.pt"))
    tracker_yaml = str(cfg_get(cfg, "yolo.tracker_yaml", "bytetrack_custom.yaml"))
    imgsz = int(cfg_get(cfg, "yolo.imgsz", 640))
    conf = float(cfg_get(cfg, "yolo.conf", 0.25))
    infer_stride = max(1, int(cfg_get(cfg, "yolo.infer_stride", 1)))
    max_det = max(1, int(cfg_get(cfg, "yolo.max_det", 10)))
    use_fp16 = bool(cfg_get(cfg, "yolo.use_fp16", True))

    logic = LogicParams(
        risk_levels={str(k): int(v) for k, v in cfg_get(cfg, "logic.risk_levels", {}).items()},
        warn_thresh=int(cfg_get(cfg, "logic.warn_thresh", 60)),
        incident_thresh=int(cfg_get(cfg, "logic.incident_thresh", 75)),
        critical_thresh=int(cfg_get(cfg, "logic.critical_thresh", 90)),
        stop_hysteresis=int(cfg_get(cfg, "logic.stop_hysteresis", 20)),
        consecutive_normal_to_stop=int(cfg_get(cfg, "logic.consecutive_normal_to_stop", 45)),
        dwell_frames={str(k): int(v) for k, v in cfg_get(cfg, "logic.dwell_frames", {}).items()},
        default_dwell=int(cfg_get(cfg, "logic.default_dwell", 6)),
        cooldown_seconds=float(cfg_get(cfg, "logic.cooldown_seconds", 3.0)),
        cooldown_overrides=set(str(x) for x in cfg_get(cfg, "logic.cooldown_overrides", ["phone"])),
        max_incident_seconds=float(cfg_get(cfg, "logic.max_incident_seconds", 30.0)),
        lost_timeout_frames=int(cfg_get(cfg, "logic.lost_timeout_frames", 30)),
    )

    window_name = str(cfg_get(cfg, "ui.window_name", "ExamGuard Dashboard"))
    show_window = bool(cfg_get(cfg, "ui.show_window", True))
    font_scale = float(cfg_get(cfg, "ui.font_scale", 0.6))
    line_thickness = int(cfg_get(cfg, "ui.line_thickness", 2))

    setup_torch_perf()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device.upper()}")
    print(f"[INFO] Loading model: {model_path}")

    model = YOLO(model_path)
    model.to(device)
    try:
        model.fuse()
    except Exception:
        pass

    cap = cv2.VideoCapture(cam_source, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:
        pass

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {real_w}x{real_h} | stride={infer_stride} | imgsz={imgsz} | conf={conf}")

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    students: Dict[int, StudentMonitor] = {}
    frame_idx = 0

    # Reuse last detection result when infer_stride > 1
    last_boxes_xyxy: Optional[np.ndarray] = None
    last_ids: Optional[np.ndarray] = None
    last_cls: Optional[np.ndarray] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now_ts = time.time()
            frame_idx += 1

            run_infer = (frame_idx % infer_stride) == 0
            if run_infer:
                # Ultralytics handles no_grad internally; we pass fp16 when on CUDA.
                results = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_yaml,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    half=(use_fp16 and device == "cuda"),
                    max_det=max_det,
                    verbose=False,
                )

                b = results[0].boxes
                if b is not None and b.id is not None:
                    # Move only what we need to CPU once.
                    last_boxes_xyxy = b.xyxy.cpu().numpy()
                    last_ids = b.id.cpu().numpy()
                    last_cls = b.cls.cpu().numpy()
                else:
                    last_boxes_xyxy, last_ids, last_cls = None, None, None

            # Draw in-place (avoid extra frame copy)
            vis = frame

            present_ids = set()

            if last_boxes_xyxy is not None and last_ids is not None and last_cls is not None:
                for box, tid, cls in zip(last_boxes_xyxy, last_ids, last_cls):
                    sid = int(tid)
                    present_ids.add(sid)

                    student = students.get(sid)
                    if student is None:
                        student = StudentMonitor(sid, logic)
                        students[sid] = student

                    student.last_seen_frame = frame_idx

                    behavior_name = normalize_behavior(model.names[int(cls)])
                    confirmed, event = student.update(now_ts, behavior_name)

                    # Event outputs (cheap)
                    if event == "WARNING":
                        logger.log("WARNING", sid, confirmed, now_ts, now_ts, student.suspicion)
                    elif event == "CRITICAL":
                        print(f"[CRITICAL] ID {sid} | behavior={confirmed} | score={student.suspicion}")
                    elif event == "INCIDENT_START":
                        print(f"[INCIDENT] Start ID {sid} | behavior={confirmed} | score={student.suspicion}")
                    elif event == "INCIDENT_END":
                        start_ts = student.incident_active_since_ts or now_ts
                        end_ts = now_ts
                        peak = student.incident_peak
                        peak_b = student.incident_peak_behavior
                        logger.log("INCIDENT", sid, peak_b, start_ts, end_ts, peak)
                        print(f"[INCIDENT] End ID {sid} | peak={peak} | behavior={peak_b}")
                        # reset peak for next incident window
                        student.incident_active_since_ts = None
                        student.incident_peak = 0
                        student.incident_peak_behavior = "normal"

                    # UI
                    x1, y1, x2, y2 = map(int, box)
                    if student.incident_active or student.suspicion >= logic.incident_thresh:
                        sev = "CRITICAL" if student.suspicion >= logic.critical_thresh else "VIOLATION"
                        draw_violation(vis, box, sev)
                    else:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), student.color, line_thickness)
                        label = f"ID:{sid} | {student.behavior} | {student.suspicion}"
                        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                        cv2.rectangle(vis, (x1, y1 - 26), (x1 + tw + 8, y1), student.color, -1)
                        cv2.putText(
                            vis,
                            label,
                            (x1 + 4, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            2,
                        )

            # End incident if lost track
            for sid, student in list(students.items()):
                if student.incident_active and sid not in present_ids:
                    if frame_idx - student.last_seen_frame > logic.lost_timeout_frames:
                        student.incident_active = False
                        end_ts = now_ts
                        start_ts = student.incident_active_since_ts or end_ts
                        logger.log("INCIDENT", sid, student.incident_peak_behavior, start_ts, end_ts, student.incident_peak)
                        print(f"[INCIDENT] Force end ID {sid} (lost track)")
                        student.incident_active_since_ts = None
                        student.incident_peak = 0
                        student.incident_peak_behavior = "normal"

                # Basic cleanup: if unseen for a while and not in incident, drop it
                if (not student.incident_active) and (frame_idx - student.last_seen_frame > 600):
                    students.pop(sid, None)

            if show_window:
                cv2.imshow(window_name, vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExamGuard Lite — optimized detection + incident logic")
    parser.add_argument("--config", default="examguard_config.json", help="Path to JSON config")
    parser.add_argument(
        "--no-write-default-config",
        action="store_true",
        help="If config is missing, do not write defaults (raise an error)",
    )
    args = parser.parse_args()

    main(config_path=args.config, write_default_config=(not args.no_write_default_config))
