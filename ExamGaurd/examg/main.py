import cv2
import numpy as np
import torch
import os
import time
import threading
import queue
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ==============================
# CONFIGURATION
# ==============================
# System
VIDEO_SOURCE = 1
CAMERA_WIDTH = 2514
CAMERA_HEIGHT = 1080

# AI Models
INFERENCE_SIZE = 640  
CONF_BEHAVIOR = 0.20
MODEL_PATH = "best.pt"
TRACKER_YAML = "bytetrack_custom.yaml"

# Proctoring Logic
SUSPICION_THRESH = 60       # Score needed to trigger an alert (0-100)
BUFFER_SIZE = 60            # Frames to keep in memory (2 seconds @ 30fps)
LOST_TIMEOUT = 30           # Frames to wait before stopping recording for a lost student
MAX_CLIP_FRAMES = 900       # Max clip length (30s)

# VIDEO OUTPUT SETTINGS
RECORDING_FPS = 30.0        

RISK_LEVELS = {
    "normal": -2,           
    "looking_left": 3,      
    "looking_right": 3,
    "looking_back": 5,      
    "phone": 10,            
    "mobile": 10
}

# Output
REPORT_DIR = "exam_reports"

# ==============================
# EVIDENCE LOGGER (THREADED)
# ==============================
class EvidenceLogger:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_path = os.path.join(REPORT_DIR, self.session_id)
        self.video_path = os.path.join(self.base_path, "video_clips")
        
        os.makedirs(self.video_path, exist_ok=True)
        
        self.csv_path = os.path.join(self.base_path, "incident_log.csv")
        with open(self.csv_path, "w") as f:
            f.write("Timestamp,Frame,ID,Behavior,Suspicion_Score,Video_File\n")
            
        # --- THREADING SETUP ---
        self.write_queue = queue.Queue()
        self.is_running = True
        self.writer_thread = threading.Thread(target=self._video_writer_worker, daemon=True)
        self.writer_thread.start()
        
        print(f"[INFO] Session started. Evidence saving to: {self.base_path}")

    def start_recording(self, student_id, width, height, fps=RECORDING_FPS):
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"ID{student_id}_violation_{timestamp}.avi"
        filepath = os.path.join(self.video_path, filename)
        
        # Send "START" command to worker thread
        self.write_queue.put({
            "type": "START",
            "sid": student_id,
            "path": filepath,
            "width": width,
            "height": height,
            "fps": fps
        })
        
        print(f"[REC] Clip queued for ID {student_id}")
        return filename

    def write_frame(self, student_id, frame):
        # Send frame to worker thread (Non-blocking)
        # We copy the frame to ensure it doesn't get modified by main loop before writing
        self.write_queue.put({
            "type": "WRITE",
            "sid": student_id,
            "frame": frame.copy() 
        })
        return 0 # We don't track length in main thread anymore

    def stop_recording(self, student_id):
        # Send "STOP" command
        self.write_queue.put({
            "type": "STOP",
            "sid": student_id
        })

    def log_incident_text(self, frame_count, student_id, behavior, suspicion, filename):
        # CSV logging is fast enough to keep in main thread
        with open(self.csv_path, "a") as f:
            f.write(f"{datetime.now()},{frame_count},{student_id},{behavior},{suspicion},{filename}\n")

    def close_all(self):
        self.is_running = False
        self.write_queue.put(None) # Sentinel to kill thread
        self.writer_thread.join()

    def _video_writer_worker(self):
        """Background thread that handles all disk I/O for videos"""
        writers = {} # Map: sid -> cv2.VideoWriter

        while self.is_running:
            try:
                task = self.write_queue.get(timeout=1)
            except queue.Empty:
                continue

            if task is None: 
                break # Exit signal

            t_type = task.get("type")
            sid = task.get("sid")

            if t_type == "START":
                if sid not in writers:
                    path = task["path"]
                    w = task["width"]
                    h = task["height"]
                    fps = task["fps"]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writers[sid] = cv2.VideoWriter(path, fourcc, fps, (w, h))

            elif t_type == "WRITE":
                if sid in writers:
                    writers[sid].write(task["frame"])

            elif t_type == "STOP":
                if sid in writers:
                    writers[sid].release()
                    del writers[sid]
                    print(f"[REC] Finished saving clip for ID {sid}")
            
            self.write_queue.task_done()

        # Cleanup
        for w in writers.values():
            w.release()

# ==============================
# STUDENT MONITOR (Logic)
# ==============================
class StudentMonitor:
    def __init__(self, student_id):
        self.id = student_id
        self.suspicion_score = 0
        self.current_behavior = "normal"
        self.color = (0, 255, 0)
        self.is_recording = False  # Track recording state
        self.last_seen = 0         # Frame count when last detected
        self.consecutive_normal = 0 # Track how long they've been good
        self.frames_recorded = 0   # Track duration to force stop

    def update(self, behavior):
        self.current_behavior = behavior
        
        if behavior == "normal":
            self.consecutive_normal += 1
        else:
            self.consecutive_normal = 0

        risk_step = RISK_LEVELS.get(behavior, 0)
        if risk_step == 0 and behavior != "normal" and behavior not in RISK_LEVELS:
             pass

        self.suspicion_score += risk_step
        self.suspicion_score = max(0, min(100, self.suspicion_score))
        
        if self.suspicion_score < 30:
            self.color = (0, 255, 0)   
        elif self.suspicion_score < SUSPICION_THRESH:
            self.color = (0, 165, 255) 
        else:
            self.color = (0, 0, 255)   

# ==============================
# ID STITCHER
# ==============================
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
            sid = None

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
                    dist = np.sqrt((cx - hist_cx)**2 + (cy - hist_cy)**2)
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
                        if curr_iou > 0.5:
                             if curr_iou > best_iou:
                                 match_ghost_active = cand_sid

                if match_iou_inactive: best_match_sid = match_iou_inactive
                elif match_dist_inactive: best_match_sid = match_dist_inactive
                elif match_ghost_active: best_match_sid = match_ghost_active

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
# MAIN SYSTEM
# ==============================
def draw_violation_highlight(frame, box, label_text):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    line_len = 30
    thick = 2
    color = (255, 255, 255)
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, thick)
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, thick)
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, thick)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, thick)

    (w, h), _ = cv2.getTextSize("VIOLATION", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(frame, (x1, y1 - 45), (x1 + w + 20, y1), (0, 0, 255), -1)
    cv2.putText(frame, "VIOLATION", (x1 + 10, y1 - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running on: {device.upper()}")
    
    print(f"[INFO] Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera Res: {real_w}x{real_h}")
    
    stitcher = IDStitcher()
    logger = EvidenceLogger()
    students = {} 
    frame_count = 0
    
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    
    cv2.namedWindow("ExamGuard Dashboard", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            clean_frame = frame.copy()
            frame_buffer.append(clean_frame)
            
            # --- A. TRACKING ---
            results = model.track(
                frame, persist=True, tracker=TRACKER_YAML,
                conf=CONF_BEHAVIOR, imgsz=INFERENCE_SIZE, device=device, verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
            else:
                boxes, track_ids, clss, confs = [], [], [], []
                
            stable_ids = stitcher.update(boxes, track_ids)
            present_sids = set()

            # --- B. PROCESS & DRAW ---
            for box, sid, cls, conf in zip(boxes, stable_ids, clss, confs):
                if sid not in students:
                    students[sid] = StudentMonitor(sid)
                
                present_sids.add(sid)
                student = students[sid]
                student.last_seen = frame_count 
                
                # UPDATE STATE FIRST
                raw_name = model.names[int(cls)]
                behavior_name = raw_name.strip().lower().replace(" ", "_")
                if "phone" in behavior_name: behavior_name = "phone"
                student.update(behavior_name)
                
                # DRAWING
                x1, y1, x2, y2 = map(int, box)
                
                if student.is_recording:
                    draw_violation_highlight(frame, box, behavior_name)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), student.color, 2)
                    header_text = f"ID:{sid} | {student.current_behavior}"
                    (w, h), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w + 10, y1), student.color, -1)
                    cv2.putText(frame, header_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    bar_width = x2 - x1
                    fill_width = int(bar_width * (student.suspicion_score / 100))
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (50,50,50), -1) 
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + fill_width, y2 + 15), student.color, -1) 

                # VIDEO RECORDING TRIGGER
                if student.suspicion_score >= SUSPICION_THRESH:
                    if not student.is_recording:
                         if behavior_name != "normal":
                            student.is_recording = True
                            student.frames_recorded = 0
                            fname = logger.start_recording(sid, real_w, real_h, fps=RECORDING_FPS)
                            
                            # Dump history (THREADED WRITE)
                            for buf_frame in frame_buffer:
                                logger.write_frame(sid, buf_frame)
                            
                            logger.log_incident_text(frame_count, sid, behavior_name, student.suspicion_score, fname)

                # Stop Condition
                if student.is_recording:
                    should_stop = False
                    student.frames_recorded += 1
                    
                    if student.suspicion_score < (SUSPICION_THRESH - 20):
                        should_stop = True
                    elif student.consecutive_normal > 45:
                        should_stop = True
                        print(f"[REC] Stopping ID {sid} (Behaved normal for 1.5s)")

                    if should_stop:
                        student.is_recording = False
                        logger.stop_recording(sid)
                    else:
                        # WRITE FRAME (THREADED)
                        logger.write_frame(sid, frame)
                        if student.frames_recorded > MAX_CLIP_FRAMES:
                             print(f"[REC] Max duration reached for ID {sid}. Cutting clip.")
                             student.is_recording = False
                             logger.stop_recording(sid)

            # --- C. FORCE STOP FOR LOST STUDENTS ---
            for sid, student in students.items():
                if student.is_recording and (sid not in present_sids):
                    if frame_count - student.last_seen > LOST_TIMEOUT:
                        print(f"[REC] Force stopping ID {sid} (Student Lost Track)")
                        logger.stop_recording(sid)
                        student.is_recording = False

            cv2.imshow("ExamGuard Dashboard", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        logger.close_all()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()