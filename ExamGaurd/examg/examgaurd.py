"""
ExamGuard — Always-Recording Dashcam Buffer + Smarter Incident Logic (Feature #1)

This file combines:
A) A dashcam-style rolling recorder (always recording to disk in short segments)
B) Correct playback speed even if inference FPS drops (fixed-FPS paced recorder)
C) Export as ONE merged mp4 per incident (ffmpeg, NVENC first then CPU fallback)
D) Overlay is included in final evidence clips (we record the rendered frame)
E) FEATURE #1: Smarter incident definition:
   1) DWELL TIME confirmation: behavior must persist N consecutive frames to count
   2) COOLDOWN: after incident ends, don't start another immediately (unless override)
   3) SEVERITY TIERS: warning / incident / critical with different actions

Why the "paced recorder" exists:
- Your inference loop may run at 10–20 FPS.
- If you write frames "as they come" but claim 30 FPS in the video header,
  playback becomes sped up (shorter than real time).
- Fix: recorder thread writes exactly 30 frames/sec and repeats the last frame if needed.

Why segments exist:
- Segments allow "save last N seconds" by copying a few files and merging.
- No burst re-encoding in the main loop -> no FPS drops during export.
"""

import cv2
import numpy as np
import torch
import os
import time
import threading
import queue
import shutil
import json
import subprocess
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional

from ultralytics import YOLO


# ==============================
# CONFIGURATION
# ==============================

# Camera / capture
VIDEO_SOURCE = 1
CAMERA_WIDTH = 2514
CAMERA_HEIGHT = 1080

# YOLO / tracking
INFERENCE_SIZE = 640
CONF_BEHAVIOR = 0.20
MODEL_PATH = "best.pt"
TRACKER_YAML = "bytetrack_custom.yaml"

# Output paths
REPORT_DIR = "exam_reports"

# ------------------------------------------------
# Recording settings (dashcam style)
# ------------------------------------------------
RECORDING_FPS = 30.0         # IMPORTANT: output playback speed target

SEGMENT_SECONDS = 2.0        # short segments => easy rolling delete & fast export slicing
ROLLING_MINUTES = 2.0        # keep last N minutes in rolling buffer

PRE_ROLL_SECONDS = 2.0       # include N seconds BEFORE incident starts
POST_ROLL_SECONDS = 2.0      # include N seconds AFTER incident ends

# If encoding full resolution is too heavy, set RECORD_SCALE < 1.0 (e.g., 0.5)
# Detection still uses full raw frame; only recording is downscaled.
RECORD_SCALE = 1.0

# OpenCV segment encoder format
FOURCC = "XVID"
SEGMENT_EXT = "avi"

# Export merged clip format
MERGED_EXT = "mp4"
DELETE_SEGMENT_COPIES_AFTER_MERGE = True

# ------------------------------------------------
# Proctoring scoring + Smarter Incident Logic (Feature #1)
# ------------------------------------------------
# Risk steps added per frame (after dwell confirmation)
RISK_LEVELS = {
    "normal": -2,
    "looking_left": 3,
    "looking_right": 3,
    "looking_back": 5,
    "phone": 10,     # "pulling out phone" -> normalize to "phone"
    "mobile": 10
}

# Severity thresholds:
# - WARNING logs only (no clip)
# - INCIDENT exports evidence clip
# - CRITICAL exports evidence clip + immediate alert print
WARN_THRESH = 60
INCIDENT_THRESH = 75
CRITICAL_THRESH = 90

# Hysteresis (prevents incident end/start flapping):
# stop incident if score < (INCIDENT_THRESH - STOP_HYSTERESIS)
STOP_HYSTERESIS = 20

# If student stays normal for this many consecutive frames, end incident
CONSECUTIVE_NORMAL_TO_STOP = 45

# Dwell time (debounce) — how many consecutive frames a behavior must appear before it counts
# Why: reduces false positives from one-frame misclassifications.
DWELL_FRAMES = {
    "phone": 3,          # phone should trigger fast
    "looking_left": 6,   # require a sustained look away
    "looking_right": 6,
    "looking_back": 8
}
DEFAULT_DWELL = 6

# Cooldown after an incident ends (seconds)
# Why: prevents back-to-back incidents caused by jitter around thresholds.
COOLDOWN_SECONDS = 3.0

# Behaviors that can override cooldown (if they occur, we can start an incident even during cooldown)
# Why: phone is high-risk; you don't want cooldown to hide it.
COOLDOWN_OVERRIDES = {"phone"}

# Maximum incident duration in REAL TIME (seconds)
# Why: inference FPS varies, so frame-count-based caps are unreliable.
MAX_INCIDENT_SECONDS = 30.0

# Queue limit for export jobs
EXPORT_QUEUE_MAX = 50


# ==============================
# STUDENT MONITOR
# ==============================
class StudentMonitor:
    """
    Holds per-student state and implements Feature #1 logic:
    - dwell confirmation
    - cooldown
    - warning/critical edge triggers
    """

    def __init__(self, student_id: int):
        self.id = student_id

        # Rolling risk score (0..100)
        self.suspicion_score = 0

        # Displayed behavior label (the "confirmed" one)
        self.current_behavior = "normal"

        # UI color (green/orange/red for non-violation display)
        self.color = (0, 255, 0)

        # Tracking bookkeeping
        self.last_seen = 0
        self.consecutive_normal = 0

        # -------------------------------
        # Dwell (debounce) state
        # -------------------------------
        # We track consecutive frames of the same raw predicted behavior.
        # Only when streak >= DWELL_FRAMES[behavior] do we accept it as "confirmed".
        self._raw_streak_behavior = "normal"
        self._raw_streak_len = 0

        # -------------------------------
        # Incident state
        # -------------------------------
        self.incident_active = False
        self.incident_start_ts: Optional[float] = None         # export range start (includes pre-roll)
        self.incident_active_since_ts: Optional[float] = None  # actual moment incident became active (no pre-roll)

        self.incident_export_end_ts: Optional[float] = None    # end+post-roll
        self.incident_finalize_ts: Optional[float] = None      # export_end + segment guard

        # Peak metadata during incident (for logging)
        self.incident_peak_suspicion = 0
        self.incident_peak_behavior = "normal"

        # -------------------------------
        # Cooldown state (Feature #1)
        # -------------------------------
        self.cooldown_until_ts: float = 0.0

        # -------------------------------
        # Warning / Critical edge triggers (Feature #1)
        # -------------------------------
        # We log warning once per "cycle" (avoid spamming every frame).
        self.warning_armed = True
        self.critical_armed = True

    # -------------------------------
    # Feature #1 — dwell confirmation
    # -------------------------------
    def confirm_behavior(self, raw_behavior: str) -> str:
        """
        Convert noisy per-frame raw behavior predictions into a confirmed behavior
        by requiring DWELL_FRAMES consecutive frames of the same non-normal behavior.

        WHY:
        - Models can flicker for 1–2 frames (especially with motion blur or partial occlusion).
        - Requiring consecutive confirmation reduces false positives dramatically.

        Rule:
        - If raw is 'normal' => confirmed 'normal' immediately.
        - If raw is not 'normal':
            - track streak length
            - confirmed is raw only after streak >= required dwell
            - otherwise treat it as 'normal' (i.e., do not penalize yet)
        """
        if raw_behavior == self._raw_streak_behavior:
            self._raw_streak_len += 1
        else:
            self._raw_streak_behavior = raw_behavior
            self._raw_streak_len = 1

        # If raw is normal, don't require dwell (normal should recover quickly)
        if raw_behavior == "normal":
            return "normal"

        req = DWELL_FRAMES.get(raw_behavior, DEFAULT_DWELL)
        if self._raw_streak_len >= req:
            return raw_behavior

        # Not confirmed yet -> treat as normal for scoring purposes
        return "normal"

    def update_score_and_color(self, confirmed_behavior: str):
        """
        Update suspicion score using confirmed behavior (after dwell filtering).

        WHY:
        - We want scoring to reflect only sustained behaviors.
        - This prevents one-frame glitches from inflating suspicion.
        """
        self.current_behavior = confirmed_behavior

        if confirmed_behavior == "normal":
            self.consecutive_normal += 1
        else:
            self.consecutive_normal = 0

        # Update suspicion score
        step = RISK_LEVELS.get(confirmed_behavior, 0)
        self.suspicion_score = max(0, min(100, self.suspicion_score + step))

        # Update UI color
        if self.suspicion_score < 30:
            self.color = (0, 255, 0)
        elif self.suspicion_score < INCIDENT_THRESH:
            self.color = (0, 165, 255)
        else:
            self.color = (0, 0, 255)


# ==============================
# ID STITCHER (your logic)
# ==============================
class IDStitcher:
    """
    Converts tracker IDs (tid) into stable IDs (sid) across short losses.
    """
    def __init__(self, max_gap=60, merge_iou=0.2, dist_thresh=400):
        self.next_sid = 1
        self.tid_to_sid = {}
        self.sid_history = {}
        self.max_gap = max_gap
        self.merge_iou = merge_iou
        self.dist_thresh = dist_thresh

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def update(self, boxes, track_ids):
        active_sids = set()
        clean_history = {}
        for sid, data in self.sid_history.items():
            data['lost_frames'] += 1
            if data['lost_frames'] <= self.max_gap:
                clean_history[sid] = data
        self.sid_history = clean_history

        result_sids = []

        for box, tid in zip(boxes, track_ids):
            tid = int(tid)

            if tid in self.tid_to_sid:
                sid = self.tid_to_sid[tid]
            else:
                best_match_sid = None
                cx, cy = self.get_center(box)
                best_iou = -1
                min_dist = float('inf')

                match_iou_inactive = None
                match_dist_inactive = None
                match_ghost_active = None

                for cand_sid, data in self.sid_history.items():
                    hist_box = data['box']
                    curr_iou = self.iou(box, hist_box)
                    hist_cx, hist_cy = self.get_center(hist_box)
                    dist = np.sqrt((cx - hist_cx) ** 2 + (cy - hist_cy) ** 2)
                    is_active = cand_sid in active_sids

                    if not is_active:
                        if curr_iou > self.merge_iou:
                            if curr_iou > best_iou:
                                best_iou = curr_iou
                                match_iou_inactive = cand_sid
                        elif dist < self.dist_thresh:
                            if dist < min_dist:
                                min_dist = dist
                                match_dist_inactive = cand_sid
                    else:
                        if curr_iou > 0.5 and curr_iou > best_iou:
                            match_ghost_active = cand_sid
                            best_iou = curr_iou

                if match_iou_inactive:
                    best_match_sid = match_iou_inactive
                elif match_dist_inactive:
                    best_match_sid = match_dist_inactive
                elif match_ghost_active:
                    best_match_sid = match_ghost_active

                if best_match_sid is not None:
                    sid = best_match_sid
                    self.tid_to_sid[tid] = sid
                else:
                    sid = self.next_sid
                    self.next_sid += 1
                    self.tid_to_sid[tid] = sid

            self.sid_history[sid] = {'box': box, 'lost_frames': 0}
            active_sids.add(sid)
            result_sids.append(sid)

        return result_sids


# ==============================
# UI HELPERS
# ==============================
def draw_violation_highlight(frame, box, severity_text="VIOLATION"):
    """
    Draw a strong red indicator for evidence clips + UI.
    This is drawn on vis_frame, which is also what we record.
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    label = severity_text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(frame, (x1, y1 - 45), (x1 + w + 20, y1), (0, 0, 255), -1)
    cv2.putText(frame, label, (x1 + 10, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


# ==============================
# DASHCAM ROLLING RECORDER (PACED FPS)
# ==============================
@dataclass
class SegmentMeta:
    path: str
    start_ts: float
    end_ts: float


class RollingSegmentRecorder:
    """
    Writes at fixed cadence (RECORDING_FPS), repeating the last frame if needed.
    This guarantees correct playback duration even when inference is slow.
    """
    def __init__(self, base_dir: str, width: int, height: int, fps: float):
        self.base_dir = base_dir
        self.width = width
        self.height = height
        self.fps = float(fps)

        self.segment_dir = os.path.join(base_dir, "_rolling_segments")
        os.makedirs(self.segment_dir, exist_ok=True)

        self.fourcc = cv2.VideoWriter_fourcc(*FOURCC)
        self.segment_seconds = float(SEGMENT_SECONDS)
        self.keep_seconds = float(ROLLING_MINUTES) * 60.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._lock = threading.Lock()
        self._segments = deque()

        self._writer = None
        self._seg_start_wall = None
        self._seg_path = None
        self._seg_index = 0

        # Export protection: don't delete segments newer than this
        self.min_keep_ts = 0.0

        # Latest frame to write (already scaled, already overlayed)
        self._latest_frame = None
        self._have_frame = False

        self._thread.start()

    def close(self):
        self._stop_event.set()
        self._thread.join()
        self._close_writer()
        self._finalize_segment_meta(end_ts=time.time())

    def submit_latest(self, frame_bgr):
        # Store latest frame for paced writer thread
        with self._lock:
            self._latest_frame = frame_bgr
            self._have_frame = True

    def get_segments_in_range(self, start_ts: float, end_ts: float) -> List[str]:
        # Return CLOSED segments overlapping the requested time range
        with self._lock:
            segs = list(self._segments)

        out = []
        for s in segs:
            if s.end_ts >= start_ts and s.start_ts <= end_ts and os.path.exists(s.path):
                out.append(s.path)

        # unique preserve order
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def _open_new_segment(self, wall_ts: float):
        self._close_writer()
        self._seg_start_wall = wall_ts
        self._seg_index += 1
        stamp = datetime.fromtimestamp(wall_ts).strftime("%H-%M-%S")
        self._seg_path = os.path.join(self.segment_dir, f"seg_{self._seg_index:06d}_{stamp}.{SEGMENT_EXT}")
        self._writer = cv2.VideoWriter(self._seg_path, self.fourcc, self.fps, (self.width, self.height))

    def _close_writer(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None

    def _finalize_segment_meta(self, end_ts: float):
        if self._seg_path is None or self._seg_start_wall is None:
            return
        meta = SegmentMeta(path=self._seg_path, start_ts=self._seg_start_wall, end_ts=end_ts)
        with self._lock:
            if os.path.exists(meta.path):
                self._segments.append(meta)

    def _cleanup_old(self, now_ts: float):
        # Delete segments outside rolling window unless protected by min_keep_ts
        cutoff = now_ts - self.keep_seconds
        with self._lock:
            while self._segments:
                s = self._segments[0]
                if s.end_ts < cutoff and s.end_ts < self.min_keep_ts:
                    self._segments.popleft()
                    try:
                        os.remove(s.path)
                    except Exception:
                        pass
                else:
                    break

    def _worker(self):
        interval = 1.0 / max(1.0, self.fps)
        next_tick = time.perf_counter()

        last_cleanup_wall = 0.0
        last_written = None

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(0.005, next_tick - now))
                continue

            # If we were paused heavily, resync schedule
            if now - next_tick > 0.5:
                next_tick = now
            else:
                next_tick += interval

            wall_ts = time.time()

            # Ensure writer exists
            if self._writer is None:
                self._open_new_segment(wall_ts)

            # Rotate segment
            if self._seg_start_wall is not None and (wall_ts - self._seg_start_wall) >= self.segment_seconds:
                self._close_writer()
                self._finalize_segment_meta(end_ts=wall_ts)
                self._open_new_segment(wall_ts)

            # Fetch latest frame (or repeat last)
            with self._lock:
                if self._have_frame and self._latest_frame is not None:
                    last_written = self._latest_frame

            if last_written is not None and self._writer is not None:
                try:
                    self._writer.write(last_written)
                except Exception:
                    pass

            # Cleanup old segments
            if wall_ts - last_cleanup_wall > 1.0:
                self._cleanup_old(wall_ts)
                last_cleanup_wall = wall_ts


# ==============================
# FFmpeg MERGE HELPERS
# ==============================
def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def write_ffmpeg_concat_list(list_path: str, segment_paths: List[str]) -> None:
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            norm = os.path.abspath(p).replace("\\", "/")
            norm = norm.replace("'", r"'\''")
            f.write(f"file '{norm}'\n")


def run_ffmpeg_merge(segment_paths: List[str], out_path: str) -> Tuple[bool, str]:
    """
    Merge segments into one MP4.
    We enforce CFR with -vf fps=RECORDING_FPS to guarantee correct timing.

    NVENC first (GPU), fallback to libx264 (CPU).
    """
    if not segment_paths:
        return False, "no_segments"
    if not ffmpeg_exists():
        return False, "ffmpeg_missing"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    list_path = out_path + "_concat.txt"
    write_ffmpeg_concat_list(list_path, segment_paths)

    creationflags = 0
    try:
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    except Exception:
        creationflags = 0

    def _run(cmd: List[str]) -> bool:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
            return True
        except Exception:
            return False

    cmd_nvenc = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-an",
        "-vf", f"fps={RECORDING_FPS}",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr",
        "-cq", "23",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    if _run(cmd_nvenc):
        try:
            os.remove(list_path)
        except Exception:
            pass
        return True, "nvenc"

    cmd_x264 = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-an",
        "-vf", f"fps={RECORDING_FPS}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    if _run(cmd_x264):
        try:
            os.remove(list_path)
        except Exception:
            pass
        return True, "libx264"

    try:
        os.remove(list_path)
    except Exception:
        pass
    return False, "ffmpeg_failed"


# ==============================
# EVIDENCE MANAGER
# ==============================
class EvidenceManager:
    """
    Handles session folders + export jobs.

    IMPORTANT DESIGN:
    - Main loop must not block on disk I/O (copying/merging videos).
    - So we queue export jobs to a background worker thread.
    """
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_path = os.path.join(REPORT_DIR, self.session_id)
        self.incidents_path = os.path.join(self.base_path, "incidents")

        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.incidents_path, exist_ok=True)

        # Unified event log (warnings + incidents)
        # Type: WARNING | INCIDENT | CRITICAL
        self.csv_path = os.path.join(self.base_path, "event_log.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("Type,StartISO,EndISO,ID,Behavior,PeakSuspicion,MergedFile,MergeMode\n")

        self.export_q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=EXPORT_QUEUE_MAX)
        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()

        self.recorder: Optional[RollingSegmentRecorder] = None

        # Export protection for rolling cleanup
        self._pending_lock = threading.Lock()
        self._pending_starts = set()

        print(f"[INFO] Session started. Evidence saving to: {self.base_path}")
        if not ffmpeg_exists():
            print("[WARN] FFmpeg not found in PATH. Merging will fail until ffmpeg is installed.")

    def attach_recorder(self, record_w: int, record_h: int, fps: float):
        self.recorder = RollingSegmentRecorder(self.base_path, record_w, record_h, fps)

    def set_min_keep_ts(self, ts: float):
        if self.recorder is not None:
            self.recorder.min_keep_ts = ts

    def submit_latest_frame(self, frame_bgr):
        if self.recorder is not None:
            self.recorder.submit_latest(frame_bgr)

    def log_warning(self, sid: int, behavior: str, score: int, ts: float):
        """
        WARNING event logging (no clip).
        Kept in main thread because it's tiny I/O (append one line).
        """
        iso = datetime.fromtimestamp(ts).isoformat()
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"WARNING,{iso},{iso},{sid},{behavior},{score},,\n")

    def add_pending_keep(self, start_ts: float):
        with self._pending_lock:
            self._pending_starts.add(start_ts)

    def remove_pending_keep(self, start_ts: float):
        with self._pending_lock:
            self._pending_starts.discard(start_ts)

    def get_pending_min_keep_ts(self, default_ts: float) -> float:
        with self._pending_lock:
            return min(self._pending_starts) if self._pending_starts else default_ts

    def queue_export(self, event_type: str, sid: int, start_ts: float, end_ts: float, behavior: str, peak_susp: int):
        """
        Enqueue INCIDENT/CRITICAL export:
        - Copy overlapping rolling segments
        - Merge into one MP4 evidence clip
        """
        if self.recorder is None:
            return

        self.add_pending_keep(start_ts)

        seg_paths = self.recorder.get_segments_in_range(start_ts, end_ts)
        job = {
            "event_type": event_type,  # INCIDENT or CRITICAL
            "sid": sid,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "behavior": behavior,
            "peak_susp": peak_susp,
            "segments": seg_paths
        }

        try:
            self.export_q.put_nowait(job)
        except queue.Full:
            print("[WARN] Export queue full. Dropping an export job.")
            self.remove_pending_keep(start_ts)

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
        self.export_q.put(None)
        self.export_thread.join()

    def _export_worker(self):
        """
        Background export worker:
        - Copies required rolling segments into incident folder
        - Merges them into one mp4
        - Deletes copied segments if merge succeeded (so you don't get many clips)
        - Logs the event in CSV
        """
        while True:
            job = self.export_q.get()
            if job is None:
                break

            event_type = job["event_type"]
            sid = job["sid"]
            start_ts = job["start_ts"]
            end_ts = job["end_ts"]
            behavior = job["behavior"]
            peak_susp = job["peak_susp"]
            segs = job["segments"]

            start_iso = datetime.fromtimestamp(start_ts).isoformat()
            end_iso = datetime.fromtimestamp(end_ts).isoformat()

            folder_name = (
                f"{event_type}_ID{sid}_"
                f"{datetime.fromtimestamp(start_ts).strftime('%H-%M-%S')}"
                f"_to_{datetime.fromtimestamp(end_ts).strftime('%H-%M-%S')}"
            )
            out_folder = os.path.join(self.incidents_path, folder_name)
            os.makedirs(out_folder, exist_ok=True)

            seg_folder = os.path.join(out_folder, "segments")
            os.makedirs(seg_folder, exist_ok=True)

            copied = []
            for p in segs:
                if not os.path.exists(p):
                    continue
                dst = os.path.join(seg_folder, os.path.basename(p))
                try:
                    shutil.copy2(p, dst)
                    copied.append(dst)
                except Exception:
                    pass

            merged_file = f"evidence_merged.{MERGED_EXT}"
            merged_path = os.path.join(out_folder, merged_file)

            ok, mode = run_ffmpeg_merge(copied, merged_path)

            manifest = {
                "type": event_type,
                "sid": sid,
                "start_iso": start_iso,
                "end_iso": end_iso,
                "behavior": behavior,
                "peak_suspicion": peak_susp,
                "merge_success": ok,
                "merge_mode": mode,
                "merged_file": merged_file if ok else None,
                "segments_used": [os.path.basename(x) for x in copied],
                "note": (
                    "Segments are used for dashcam-style rolling buffering (cheap pre-roll). "
                    "Final clip includes overlays because recorder records vis_frame. "
                    "Playback speed fixed by paced recorder + CFR fps filter in ffmpeg."
                )
            }
            with open(os.path.join(out_folder, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            with open(self.csv_path, "a", encoding="utf-8") as f:
                merged_name = merged_file if ok else ""
                f.write(f"{event_type},{start_iso},{end_iso},{sid},{behavior},{peak_susp},{merged_name},{mode}\n")

            if ok and DELETE_SEGMENT_COPIES_AFTER_MERGE:
                for p in copied:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                try:
                    os.rmdir(seg_folder)
                except Exception:
                    pass
                print(f"[SAVE] {event_type} merged for ID {sid} -> {merged_path} (segment copies deleted)")
            else:
                if ok:
                    print(f"[SAVE] {event_type} merged for ID {sid} -> {merged_path} (segment copies kept)")
                else:
                    print(f"[SAVE] {event_type} saved but merge failed (mode={mode}). Segments kept at: {seg_folder}")

            self.remove_pending_keep(start_ts)


# ==============================
# MAIN LOOP
# ==============================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running on: {device.upper()}")

    print(f"[INFO] Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(device)

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Reduce latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera Res: {real_w}x{real_h}")

    # Recording resolution (may be downscaled)
    rec_w = int(real_w * RECORD_SCALE)
    rec_h = int(real_h * RECORD_SCALE)
    if rec_w % 2 == 1:
        rec_w -= 1
    if rec_h % 2 == 1:
        rec_h -= 1
    print(f"[INFO] Recording Res: {rec_w}x{rec_h} @ {RECORDING_FPS}fps (scale={RECORD_SCALE})")

    stitcher = IDStitcher()
    evidence = EvidenceManager()
    evidence.attach_recorder(rec_w, rec_h, RECORDING_FPS)

    students: dict[int, StudentMonitor] = {}
    frame_count = 0

    cv2.namedWindow("ExamGuard Dashboard", cv2.WINDOW_NORMAL)

    def compute_min_keep_ts(now_ts: float) -> float:
        """
        Prevent rolling cleanup from deleting segments needed for:
        - active incidents
        - post-roll waiting incidents
        - pending exports
        """
        starts = []
        for s in students.values():
            if s.incident_active and s.incident_start_ts is not None:
                starts.append(s.incident_start_ts)
            if (not s.incident_active) and (s.incident_finalize_ts is not None) and (s.incident_start_ts is not None):
                starts.append(s.incident_start_ts)

        pending_min = evidence.get_pending_min_keep_ts(default_ts=now_ts)
        starts.append(pending_min)

        return min(starts) if starts else (now_ts - (ROLLING_MINUTES * 60.0))

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            now_ts = time.time()
            frame_count += 1

            # We draw onto this and also record it, so evidence clips show overlays
            vis_frame = raw_frame.copy()

            # ------------------------------
            # Inference (on raw frame)
            # ------------------------------
            results = model.track(
                raw_frame,
                persist=True,
                tracker=TRACKER_YAML,
                conf=CONF_BEHAVIOR,
                imgsz=INFERENCE_SIZE,
                device=device,
                verbose=False
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
            else:
                boxes, track_ids, clss = [], [], []

            stable_ids = stitcher.update(boxes, track_ids)
            present_sids = set()

            # ------------------------------
            # Process detections
            # ------------------------------
            for box, sid, cls in zip(boxes, stable_ids, clss):
                if sid not in students:
                    students[sid] = StudentMonitor(sid)

                present_sids.add(sid)
                student = students[sid]
                student.last_seen = frame_count

                # Map class -> behavior string
                raw_name = model.names[int(cls)]
                raw_behavior = raw_name.strip().lower().replace(" ", "_")

                # Normalize phone-like names to "phone"
                if "phone" in raw_behavior or "mobile" in raw_behavior:
                    raw_behavior = "phone"

                # ------------------------------
                # FEATURE #1 — Dwell confirmation
                # ------------------------------
                confirmed_behavior = student.confirm_behavior(raw_behavior)

                # Update suspicion score based on confirmed behavior
                student.update_score_and_color(confirmed_behavior)

                # Peak tracking (use confirmed behavior so logs match scoring)
                if student.suspicion_score > student.incident_peak_suspicion:
                    student.incident_peak_suspicion = student.suspicion_score
                    student.incident_peak_behavior = confirmed_behavior

                # ------------------------------
                # FEATURE #1 — Warning tier (log only)
                # ------------------------------
                # We only want one warning per "cycle", not spam every frame.
                # Reset the arm when score drops meaningfully below WARN_THRESH.
                if student.suspicion_score < (WARN_THRESH - 10):
                    student.warning_armed = True

                if student.warning_armed and student.suspicion_score >= WARN_THRESH and confirmed_behavior != "normal":
                    evidence.log_warning(sid, confirmed_behavior, student.suspicion_score, now_ts)
                    student.warning_armed = False

                # ------------------------------
                # FEATURE #1 — Critical tier (alert)
                # ------------------------------
                # Same anti-spam approach as warnings.
                if student.suspicion_score < (CRITICAL_THRESH - 10):
                    student.critical_armed = True

                if student.critical_armed and student.suspicion_score >= CRITICAL_THRESH and confirmed_behavior != "normal":
                    print(f"[CRITICAL] ID {sid} | behavior={confirmed_behavior} | score={student.suspicion_score}")
                    student.critical_armed = False

                # ------------------------------
                # If we're in post-roll window and "re-offend", continue the same incident
                # WHY:
                # - avoids chopping one long event into multiple clips separated by 1–2 seconds
                # - keeps evidence cohesive for reviewers
                # ------------------------------
                if student.incident_export_end_ts is not None and now_ts <= student.incident_export_end_ts:
                    if student.suspicion_score >= INCIDENT_THRESH and confirmed_behavior != "normal":
                        student.incident_active = True
                        student.incident_export_end_ts = None
                        student.incident_finalize_ts = None
                        student.cooldown_until_ts = 0.0  # cancel cooldown if we resumed
                        if student.incident_active_since_ts is None:
                            student.incident_active_since_ts = now_ts

                # ------------------------------
                # Incident start decision (tier=INCIDENT/CRITICAL)
                # ------------------------------
                # We only export clips for score >= INCIDENT_THRESH.
                # Also respect cooldown unless behavior is an override (phone).
                in_cooldown = now_ts < student.cooldown_until_ts
                cooldown_override = confirmed_behavior in COOLDOWN_OVERRIDES

                if (student.suspicion_score >= INCIDENT_THRESH and
                        (not student.incident_active) and
                        student.incident_finalize_ts is None and
                        confirmed_behavior != "normal" and
                        ((not in_cooldown) or cooldown_override)):

                    student.incident_active = True
                    student.incident_active_since_ts = now_ts
                    student.incident_start_ts = now_ts - PRE_ROLL_SECONDS
                    student.incident_export_end_ts = None
                    student.incident_finalize_ts = None
                    print(f"[INCIDENT] Start ID {sid} | behavior={confirmed_behavior} | score={student.suspicion_score}")

                # ------------------------------
                # Incident stop decision
                # ------------------------------
                if student.incident_active:
                    should_end = False

                    # Stop if score dropped sufficiently below incident threshold
                    if student.suspicion_score < (INCIDENT_THRESH - STOP_HYSTERESIS):
                        should_end = True

                    # Or if they've been normal long enough
                    elif student.consecutive_normal > CONSECUTIVE_NORMAL_TO_STOP:
                        should_end = True

                    # Or if incident has been active too long in real time
                    if student.incident_active_since_ts is not None:
                        if (now_ts - student.incident_active_since_ts) > MAX_INCIDENT_SECONDS:
                            should_end = True
                            print(f"[INCIDENT] Max duration reached for ID {sid}. Ending incident.")

                    # If ending, set post-roll and finalize times once (don’t re-trigger every frame)
                    if should_end and student.incident_export_end_ts is None:
                        student.incident_active = False

                        export_end = now_ts + POST_ROLL_SECONDS
                        student.incident_export_end_ts = export_end

                        # Segment guard: wait one full segment so the last required segment closes
                        student.incident_finalize_ts = export_end + SEGMENT_SECONDS

                        # Cooldown starts now
                        student.cooldown_until_ts = now_ts + COOLDOWN_SECONDS

                        print(f"[INCIDENT] End ID {sid} | exporting after ~{POST_ROLL_SECONDS + SEGMENT_SECONDS:.1f}s")

                # ------------------------------
                # Draw overlay / boxes onto vis_frame
                # ------------------------------
                # We keep violation overlay during post-roll too, so the tail of evidence still looks like "violation time".
                show_violation = student.incident_active or (
                    student.incident_export_end_ts is not None and now_ts <= student.incident_export_end_ts
                )

                x1, y1, x2, y2 = map(int, box)

                if show_violation:
                    # Show CRITICAL text if score is in critical band (helps reviewers)
                    sev = "CRITICAL" if student.suspicion_score >= CRITICAL_THRESH else "VIOLATION"
                    draw_violation_highlight(vis_frame, box, severity_text=sev)
                else:
                    # Normal UI view
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), student.color, 2)
                    header_text = f"ID:{sid} | {student.current_behavior} | {student.suspicion_score}"
                    (w, h), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (x1, y1 - 30), (x1 + w + 10, y1), student.color, -1)
                    cv2.putText(vis_frame, header_text, (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ------------------------------
            # Force end incident if student disappears too long
            # (helps if tracker loses them while incident is active)
            # ------------------------------
            for sid, student in students.items():
                if student.incident_active and (sid not in present_sids):
                    # If we haven't seen them for LOST_TIMEOUT frames, end incident
                    if frame_count - student.last_seen > 30:
                        student.incident_active = False
                        export_end = now_ts + POST_ROLL_SECONDS
                        student.incident_export_end_ts = export_end
                        student.incident_finalize_ts = export_end + SEGMENT_SECONDS
                        student.cooldown_until_ts = now_ts + COOLDOWN_SECONDS
                        print(f"[INCIDENT] Force end ID {sid} (lost track)")

            # ------------------------------
            # Finalize exports: when finalize_ts reached, queue job
            # ------------------------------
            for sid, student in students.items():
                if (not student.incident_active) and (student.incident_finalize_ts is not None):
                    if now_ts >= student.incident_finalize_ts:
                        start_ts = student.incident_start_ts if student.incident_start_ts else (now_ts - PRE_ROLL_SECONDS)
                        end_ts = student.incident_export_end_ts if student.incident_export_end_ts else now_ts

                        # Determine export severity from peak score
                        event_type = "CRITICAL" if student.incident_peak_suspicion >= CRITICAL_THRESH else "INCIDENT"

                        evidence.queue_export(
                            event_type=event_type,
                            sid=sid,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            behavior=student.incident_peak_behavior,
                            peak_susp=student.incident_peak_suspicion
                        )

                        # Reset incident metadata
                        student.incident_finalize_ts = None
                        student.incident_export_end_ts = None
                        student.incident_start_ts = None
                        student.incident_active_since_ts = None
                        student.incident_peak_suspicion = 0
                        student.incident_peak_behavior = "normal"

            # ------------------------------
            # Protect segments needed for active/pending events from rolling deletion
            # ------------------------------
            evidence.set_min_keep_ts(compute_min_keep_ts(now_ts))

            # ------------------------------
            # Record the overlayed frame (vis_frame)
            # ------------------------------
            if rec_w != real_w or rec_h != real_h:
                # resize creates a new array (safe to hand off)
                rec_frame = cv2.resize(vis_frame, (rec_w, rec_h), interpolation=cv2.INTER_AREA)
            else:
                rec_frame = vis_frame

            evidence.submit_latest_frame(rec_frame)

            # Display
            cv2.imshow("ExamGuard Dashboard", vis_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        evidence.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
