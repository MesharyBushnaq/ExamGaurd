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

VIDEO_SOURCE = 1
CAMERA_WIDTH = 2514
CAMERA_HEIGHT = 1080
INFERENCE_SIZE = 640
CONF_BEHAVIOR = 0.20
MODEL_PATH = "best.pt"
TRACKER_YAML = "bytetrack_custom.yaml"
REPORT_DIR = "exam_reports"
RECORDING_FPS = 30.0         
SEGMENT_SECONDS = 2.0        
ROLLING_MINUTES = 2.0        
PRE_ROLL_SECONDS = 2.0       
POST_ROLL_SECONDS = 1.0      
RECORD_SCALE = 1.0
FOURCC = "XVID"
SEGMENT_EXT = "avi"
MERGED_EXT = "mp4"
DELETE_SEGMENT_COPIES_AFTER_MERGE = True

RISK_LEVELS = {
    "normal": -2,
    "looking_left": 3,
    "looking_right": 3,
    "looking_back": 5,
    "phone": 10,     
    "mobile": 10
}

WARN_THRESH = 60
INCIDENT_THRESH = 75
CRITICAL_THRESH = 90
STOP_HYSTERESIS = 10
CONSECUTIVE_NORMAL_TO_STOP = 45

DWELL_FRAMES = {
    "phone": 3,          
    "looking_left": 6,   
    "looking_right": 6,
    "looking_back": 8
}
DEFAULT_DWELL = 6
COOLDOWN_SECONDS = 0.5
COOLDOWN_OVERRIDES = {"phone"}
MAX_INCIDENT_SECONDS = 30.0
EXPORT_QUEUE_MAX = 50

class StudentMonitor:
    def __init__(self, student_id: int):
        self.id = student_id
        self.suspicion_score = 0
        self.current_behavior = "normal"
        self.color = (0, 255, 0)
        self.last_seen = 0
        self.consecutive_normal = 0
        self._raw_streak_behavior = "normal"
        self._raw_streak_len = 0
        self.incident_active = False
        self.incident_start_ts: Optional[float] = None         
        self.incident_active_since_ts: Optional[float] = None  
        self.incident_export_end_ts: Optional[float] = None    
        self.incident_finalize_ts: Optional[float] = None      
        self.incident_peak_suspicion = 0
        self.incident_peak_behavior = "normal"
        self.cooldown_until_ts: float = 0.0
        self.warning_armed = True
        self.critical_armed = True

    def confirm_behavior(self, raw_behavior: str) -> str:
        if raw_behavior == self._raw_streak_behavior:
            self._raw_streak_len += 1
        else:
            self._raw_streak_behavior = raw_behavior
            self._raw_streak_len = 1

        if raw_behavior == "normal":
            return "normal"

        req = DWELL_FRAMES.get(raw_behavior, DEFAULT_DWELL)
        if self._raw_streak_len >= req:
            return raw_behavior

        return "normal"

    def update_score_and_color(self, confirmed_behavior: str):
        self.current_behavior = confirmed_behavior
        if confirmed_behavior == "normal":
            self.consecutive_normal += 1
        else:
            self.consecutive_normal = 0

        step = RISK_LEVELS.get(confirmed_behavior, 0)
        self.suspicion_score = max(0, min(100, self.suspicion_score + step))

        if self.suspicion_score < 30:
            self.color = (0, 255, 0)
        elif self.suspicion_score < INCIDENT_THRESH:
            self.color = (0, 165, 255)
        else:
            self.color = (0, 0, 255)

class IDStitcher:
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

def draw_violation_highlight(frame, box, severity_text="VIOLATION"):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    label = severity_text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(frame, (x1, y1 - 45), (x1 + w + 20, y1), (0, 0, 255), -1)
    cv2.putText(frame, label, (x1 + 10, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

@dataclass
class SegmentMeta:
    path: str
    start_ts: float
    end_ts: float

class RollingSegmentRecorder:
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
        self.min_keep_ts = 0.0
        self._latest_frame = None
        self._have_frame = False
        self._thread.start()

    def close(self):
        self._stop_event.set()
        self._thread.join()
        self._close_writer()
        self._finalize_segment_meta(end_ts=time.time())

    def submit_latest(self, frame_bgr):
        with self._lock:
            self._latest_frame = frame_bgr
            self._have_frame = True

    def get_segments_in_range(self, start_ts: float, end_ts: float) -> List[str]:
        with self._lock:
            segs = list(self._segments)
        out = []
        for s in segs:
            if s.end_ts >= start_ts and s.start_ts <= end_ts and os.path.exists(s.path):
                out.append(s.path)
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
            if now - next_tick > 0.5:
                next_tick = now
            else:
                next_tick += interval
            wall_ts = time.time()
            if self._writer is None:
                self._open_new_segment(wall_ts)
            if self._seg_start_wall is not None and (wall_ts - self._seg_start_wall) >= self.segment_seconds:
                self._close_writer()
                self._finalize_segment_meta(end_ts=wall_ts)
                self._open_new_segment(wall_ts)
            with self._lock:
                if self._have_frame and self._latest_frame is not None:
                    last_written = self._latest_frame
            if last_written is not None and self._writer is not None:
                try:
                    self._writer.write(last_written)
                except Exception:
                    pass
            if wall_ts - last_cleanup_wall > 1.0:
                self._cleanup_old(wall_ts)
                last_cleanup_wall = wall_ts

def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def write_ffmpeg_concat_list(list_path: str, segment_paths: List[str]) -> None:
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            norm = os.path.abspath(p).replace("\\", "/")
            norm = norm.replace("'", r"'\''")
            f.write(f"file '{norm}'\n")

def run_ffmpeg_merge(segment_paths: List[str], out_path: str) -> Tuple[bool, str]:
    if not segment_paths:
        return False, "no_segments"
    if not ffmpeg_exists():
        return False, "ffmpeg_missing"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    list_path = out_path + "_concat.txt"
    write_ffmpeg_concat_list(list_path, segment_paths)
    creationflags = 0
    try:
        creationflags = subprocess.CREATE_NO_WINDOW
    except Exception:
        creationflags = 0
    def _run(cmd: List[str]) -> bool:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
            return True
        except Exception:
            return False
    cmd_nvenc = [
        "ffmpeg", "-y", "-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", list_path,
        "-an", "-vf", f"fps={RECORDING_FPS}", "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
        "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path
    ]
    if _run(cmd_nvenc):
        try: os.remove(list_path)
        except Exception: pass
        return True, "nvenc"
    cmd_x264 = [
        "ffmpeg", "-y", "-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", list_path,
        "-an", "-vf", f"fps={RECORDING_FPS}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path
    ]
    if _run(cmd_x264):
        try: os.remove(list_path)
        except Exception: pass
        return True, "libx264"
    try: os.remove(list_path)
    except Exception: pass
    return False, "ffmpeg_failed"

class EvidenceManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_path = os.path.join(REPORT_DIR, self.session_id)
        self.incidents_path = os.path.join(self.base_path, "incidents")
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.incidents_path, exist_ok=True)
        self.csv_path = os.path.join(self.base_path, "event_log.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("Type,StartISO,EndISO,ID,Behavior,PeakSuspicion,MergedFile,MergeMode\n")
        self.export_q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=EXPORT_QUEUE_MAX)
        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        self.recorder: Optional[RollingSegmentRecorder] = None
        self._pending_lock = threading.Lock()
        self._pending_starts = set()
        print(f"[INFO] Session started. Evidence saving to: {self.base_path}")

    def attach_recorder(self, record_w: int, record_h: int, fps: float):
        self.recorder = RollingSegmentRecorder(self.base_path, record_w, record_h, fps)

    def set_min_keep_ts(self, ts: float):
        if self.recorder is not None:
            self.recorder.min_keep_ts = ts

    def submit_latest_frame(self, frame_bgr):
        if self.recorder is not None:
            self.recorder.submit_latest(frame_bgr)

    def log_warning(self, sid: int, behavior: str, score: int, ts: float):
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
        if self.recorder is None:
            return
        self.add_pending_keep(start_ts)
        seg_paths = self.recorder.get_segments_in_range(start_ts, end_ts)
        job = {
            "event_type": event_type,
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
            folder_name = (f"{event_type}_ID{sid}_{datetime.fromtimestamp(start_ts).strftime('%H-%M-%S')}"
                           f"_to_{datetime.fromtimestamp(end_ts).strftime('%H-%M-%S')}")
            out_folder = os.path.join(self.incidents_path, folder_name)
            os.makedirs(out_folder, exist_ok=True)
            seg_folder = os.path.join(out_folder, "segments")
            os.makedirs(seg_folder, exist_ok=True)
            copied = []
            for p in segs:
                if not os.path.exists(p): continue
                dst = os.path.join(seg_folder, os.path.basename(p))
                try:
                    shutil.copy2(p, dst)
                    copied.append(dst)
                except Exception: pass
            merged_file = f"evidence_merged.{MERGED_EXT}"
            merged_path = os.path.join(out_folder, merged_file)
            ok, mode = run_ffmpeg_merge(copied, merged_path)
            manifest = {
                "type": event_type, "sid": sid, "start_iso": start_iso, "end_iso": end_iso,
                "behavior": behavior, "peak_suspicion": peak_susp, "merge_success": ok,
                "merge_mode": mode, "merged_file": merged_file if ok else None,
                "segments_used": [os.path.basename(x) for x in copied]
            }
            with open(os.path.join(out_folder, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            with open(self.csv_path, "a", encoding="utf-8") as f:
                merged_name = merged_file if ok else ""
                f.write(f"{event_type},{start_iso},{end_iso},{sid},{behavior},{peak_susp},{merged_name},{mode}\n")
            if ok and DELETE_SEGMENT_COPIES_AFTER_MERGE:
                for p in copied:
                    try: os.remove(p)
                    except Exception: pass
                try: os.rmdir(seg_folder)
                except Exception: pass
                print(f"[SAVE] {event_type} merged for ID {sid} -> {merged_path}")
            else:
                print(f"[SAVE] {event_type} result: ok={ok}, mode={mode}")
            self.remove_pending_keep(start_ts)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running on: {device.upper()}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rec_w = int(real_w * RECORD_SCALE)
    rec_h = int(real_h * RECORD_SCALE)
    if rec_w % 2 == 1: rec_w -= 1
    if rec_h % 2 == 1: rec_h -= 1
    stitcher = IDStitcher()
    evidence = EvidenceManager()
    evidence.attach_recorder(rec_w, rec_h, RECORDING_FPS)
    students: dict[int, StudentMonitor] = {}
    frame_count = 0
    cv2.namedWindow("ExamGuard Dashboard", cv2.WINDOW_NORMAL)
    def compute_min_keep_ts(now_ts: float) -> float:
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
            if not ret: break
            now_ts = time.time()
            frame_count += 1
            vis_frame = raw_frame.copy()
            results = model.track(raw_frame, persist=True, tracker=TRACKER_YAML, conf=CONF_BEHAVIOR, imgsz=INFERENCE_SIZE, device=device, verbose=False)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
            else:
                boxes, track_ids, clss = [], [], []
            stable_ids = stitcher.update(boxes, track_ids)
            present_sids = set()
            for box, sid, cls in zip(boxes, stable_ids, clss):
                if sid not in students: students[sid] = StudentMonitor(sid)
                present_sids.add(sid)
                student = students[sid]
                student.last_seen = frame_count
                raw_name = model.names[int(cls)]
                raw_behavior = raw_name.strip().lower().replace(" ", "_")
                if "phone" in raw_behavior or "mobile" in raw_behavior:
                    raw_behavior = "phone"
                confirmed_behavior = student.confirm_behavior(raw_behavior)
                student.update_score_and_color(confirmed_behavior)
                if student.suspicion_score > student.incident_peak_suspicion:
                    student.incident_peak_suspicion = student.suspicion_score
                    student.incident_peak_behavior = confirmed_behavior
                if student.suspicion_score < (WARN_THRESH - 10):
                    student.warning_armed = True
                if student.warning_armed and student.suspicion_score >= WARN_THRESH and confirmed_behavior != "normal":
                    evidence.log_warning(sid, confirmed_behavior, student.suspicion_score, now_ts)
                    student.warning_armed = False
                if student.suspicion_score < (CRITICAL_THRESH - 10):
                    student.critical_armed = True
                if student.critical_armed and student.suspicion_score >= CRITICAL_THRESH and confirmed_behavior != "normal":
                    print(f"[CRITICAL] ID {sid} | behavior={confirmed_behavior} | score={student.suspicion_score}")
                    student.critical_armed = False
                if student.incident_export_end_ts is not None and now_ts <= student.incident_export_end_ts:
                    if student.suspicion_score >= INCIDENT_THRESH and confirmed_behavior != "normal":
                        student.incident_active = True
                        student.incident_export_end_ts = None
                        student.incident_finalize_ts = None
                        student.cooldown_until_ts = 0.0
                        if student.incident_active_since_ts is None:
                            student.incident_active_since_ts = now_ts
                in_cooldown = now_ts < student.cooldown_until_ts
                cooldown_override = confirmed_behavior in COOLDOWN_OVERRIDES
                if (student.suspicion_score >= INCIDENT_THRESH and (not student.incident_active) and student.incident_finalize_ts is None and confirmed_behavior != "normal" and ((not in_cooldown) or cooldown_override)):
                    student.incident_active = True
                    student.incident_active_since_ts = now_ts
                    student.incident_start_ts = now_ts - PRE_ROLL_SECONDS
                    student.incident_export_end_ts = None
                    student.incident_finalize_ts = None
                    print(f"[INCIDENT] Start ID {sid} | behavior={confirmed_behavior}")
                if student.incident_active:
                    should_end = False
                    if student.suspicion_score < (INCIDENT_THRESH - STOP_HYSTERESIS):
                        should_end = True
                    elif student.consecutive_normal > CONSECUTIVE_NORMAL_TO_STOP:
                        should_end = True
                    if student.incident_active_since_ts is not None:
                        if (now_ts - student.incident_active_since_ts) > MAX_INCIDENT_SECONDS:
                            should_end = True
                    if should_end and student.incident_export_end_ts is None:
                        student.incident_active = False
                        export_end = now_ts + POST_ROLL_SECONDS
                        student.incident_export_end_ts = export_end
                        student.incident_finalize_ts = export_end + SEGMENT_SECONDS
                        student.cooldown_until_ts = now_ts + COOLDOWN_SECONDS
                show_violation = student.incident_active or (student.incident_export_end_ts is not None and now_ts <= student.incident_export_end_ts)
                x1, y1, x2, y2 = map(int, box)
                if show_violation:
                    sev = "CRITICAL" if student.suspicion_score >= CRITICAL_THRESH else "VIOLATION"
                    draw_violation_highlight(vis_frame, box, severity_text=sev)
                else:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), student.color, 2)
                    header_text = f"ID:{sid} | {student.current_behavior} | {student.suspicion_score}"
                    (w, h), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (x1, y1 - 30), (x1 + w + 10, y1), student.color, -1)
                    cv2.putText(vis_frame, header_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for sid, student in students.items():
                if student.incident_active and (sid not in present_sids):
                    if frame_count - student.last_seen > 30:
                        student.incident_active = False
                        export_end = now_ts + POST_ROLL_SECONDS
                        student.incident_export_end_ts = export_end
                        student.incident_finalize_ts = export_end + SEGMENT_SECONDS
                        student.cooldown_until_ts = now_ts + COOLDOWN_SECONDS
            for sid, student in students.items():
                if (not student.incident_active) and (student.incident_finalize_ts is not None):
                    if now_ts >= student.incident_finalize_ts:
                        start_ts = student.incident_start_ts if student.incident_start_ts else (now_ts - PRE_ROLL_SECONDS)
                        end_ts = student.incident_export_end_ts if student.incident_export_end_ts else now_ts
                        event_type = "CRITICAL" if student.incident_peak_suspicion >= CRITICAL_THRESH else "INCIDENT"
                        evidence.queue_export(event_type=event_type, sid=sid, start_ts=start_ts, end_ts=end_ts, behavior=student.incident_peak_behavior, peak_susp=student.incident_peak_suspicion)
                        student.incident_finalize_ts = None
                        student.incident_export_end_ts = None
                        student.incident_start_ts = None
                        student.incident_active_since_ts = None
                        student.incident_peak_suspicion = 0
                        student.incident_peak_behavior = "normal"
            evidence.set_min_keep_ts(compute_min_keep_ts(now_ts))
            if rec_w != real_w or rec_h != real_h:
                rec_frame = cv2.resize(vis_frame, (rec_w, rec_h), interpolation=cv2.INTER_AREA)
            else:
                rec_frame = vis_frame
            evidence.submit_latest_frame(rec_frame)
            cv2.imshow("ExamGuard Dashboard", vis_frame)
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        cap.release()
        evidence.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()