"""
main.py — FastAPI WebSocket Server for Real-Time Surveillance Pipeline

This is the central state manager and WebSocket endpoint. It:
  1) Receives video frames from a client over WebSocket.
  2) Runs YOLOv11 pose estimation + object detection.
  3) Tracks persons with ByteTrack (via supervision).
  4) Maintains a rolling 15-frame history per tracked person.
  5) Checks proximity between all tracked person pairs.
  6) Runs advanced violence detection (Jerk + Momentum Transfer).
  7) Runs fall detection on every individual.
  8) Returns annotated results + alerts over WebSocket as JSON.

Architecture:
  Client (browser/camera) → WebSocket → FastAPI → YOLO → ByteTrack
  → action_math (Kinematics) → Alerts → WebSocket → Client
"""

import time
import json
import base64
import asyncio
from collections import defaultdict, deque
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.action_math import (
    detect_advanced_violence,
    detect_fall,
    extract_keypoint,
)

# ============================================================
# CONFIGURABLE CONSTANTS
# ============================================================

# Maximum number of frames of keypoint history to retain per person.
# We need at least 5 for 3rd-order derivatives (jerk), but 15 gives
# us a comfortable window for smoothing + multi-frame analysis.
HISTORY_LENGTH = 15

# Distance in pixels between two persons' centroids to trigger
# proximity-based violence analysis. Only pairs within this radius
# are checked — saves compute by skipping distant people.
PROXIMITY_THRESHOLD = 200.0

# Cooldown in seconds between violence alerts for the same pair.
# Prevents flooding the client with 30 alerts/sec during a fight.
VIOLENCE_ALERT_COOLDOWN = 2.0

# Cooldown in seconds between fall alerts for the same person.
FALL_ALERT_COOLDOWN = 3.0

# YOLO confidence threshold for pose estimation
POSE_CONFIDENCE = 0.5

# YOLO keypoint index map (COCO 17-point format)
# These map human body parts to array indices in YOLO pose output.
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# ============================================================
# APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="Surveillance Pipeline API",
    description="Real-time violence & fall detection via WebSocket",
    version="1.0.0",
)

# Allow CORS for browser-based frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATE MANAGEMENT
# ============================================================

# Per-person keypoint history, keyed by ByteTrack tracker_id.
# Each entry is a dict of deques holding (x, y) tuples:
#   person_states[tracker_id] = {
#       'wrist_R': deque([(x,y), ...], maxlen=15),
#       'wrist_L': deque([(x,y), ...], maxlen=15),
#       'head':    deque([(x,y), ...], maxlen=15),
#       'shoulder_L': deque(...),
#       'shoulder_R': deque(...),
#       'hip_L':   deque(...),
#       'hip_R':   deque(...),
#       'centroid': (x, y),  # most recent centroid for proximity
#   }
person_states: Dict[int, Dict[str, Any]] = defaultdict(
    lambda: {
        'wrist_R': deque(maxlen=HISTORY_LENGTH),
        'wrist_L': deque(maxlen=HISTORY_LENGTH),
        'head': deque(maxlen=HISTORY_LENGTH),
        'shoulder_L': deque(maxlen=HISTORY_LENGTH),
        'shoulder_R': deque(maxlen=HISTORY_LENGTH),
        'hip_L': deque(maxlen=HISTORY_LENGTH),
        'hip_R': deque(maxlen=HISTORY_LENGTH),
        'centroid': (0.0, 0.0),
        'fall_streak': 0,
        'is_fallen': False,
        'bbox_ar': 1.0,
        'bbox_width': 1.0,
        'bbox_height': 1.0,
    }
)

# Cooldown timestamps: maps frozenset({id_A, id_B}) → last_alert_time
violence_cooldowns: Dict[frozenset, float] = {}

# Cooldown timestamps: maps tracker_id → last_fall_alert_time
fall_cooldowns: Dict[int, float] = {}

# Models will be loaded lazily on first WebSocket connection
pose_model = None
detection_model = None
byte_tracker = None


# ============================================================
# MODEL LOADING (Lazy Initialization)
# ============================================================
def load_models():
    """
    Lazily loads YOLO models and ByteTrack tracker on first use.
    This avoids slow startup and allows the server to boot even
    if model files are not yet present (fails on first request
    with a clear error instead of on import).
    """
    global pose_model, detection_model, byte_tracker

    if pose_model is None:
        try:
            from ultralytics import YOLO
            import supervision as sv

            # Pose model for keypoint extraction
            # TODO: Update path to your trained .pt or .onnx model
            pose_model = YOLO('models/yolo11n-pose.pt')
            print("[*] Pose model loaded successfully.")

            # Detection model for fire/weapon detection
            # TODO: Update path after training the merged dataset
            # detection_model = YOLO('models/best.onnx')
            # print("[*] Detection model loaded successfully.")

            # ByteTrack tracker from supervision library
            byte_tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            print("[*] ByteTrack tracker initialized.")

        except Exception as e:
            print(f"[!] Error loading models: {e}")
            raise


# ============================================================
# KEYPOINT HISTORY UPDATE
# ============================================================
def update_person_state(tracker_id: int, keypoints: np.ndarray):
    """
    Extracts relevant keypoints from a YOLO pose result and appends
    them to the rolling history deque for the given tracked person.

    If a keypoint is undetected (confidence too low), we append (0, 0)
    which will be handled by the _fill_missing() function in action_math.

    Args:
        tracker_id: Unique ID from ByteTrack.
        keypoints:  NumPy array of shape (17, 3) — [x, y, conf] per keypoint.
    """
    state = person_states[tracker_id]

    # --- Extract and store each relevant keypoint ---
    # Head (nose)
    head = extract_keypoint(keypoints, KP_NOSE)
    state['head'].append(head if head else (0.0, 0.0))

    # Right wrist (primary strike hand)
    wr = extract_keypoint(keypoints, KP_RIGHT_WRIST)
    state['wrist_R'].append(wr if wr else (0.0, 0.0))

    # Left wrist (fallback)
    wl = extract_keypoint(keypoints, KP_LEFT_WRIST)
    state['wrist_L'].append(wl if wl else (0.0, 0.0))

    # Shoulders (for fall detection torso angle)
    sl = extract_keypoint(keypoints, KP_LEFT_SHOULDER)
    state['shoulder_L'].append(sl if sl else (0.0, 0.0))

    sr = extract_keypoint(keypoints, KP_RIGHT_SHOULDER)
    state['shoulder_R'].append(sr if sr else (0.0, 0.0))

    # Hips (for fall detection torso angle)
    hl = extract_keypoint(keypoints, KP_LEFT_HIP)
    state['hip_L'].append(hl if hl else (0.0, 0.0))

    hr = extract_keypoint(keypoints, KP_RIGHT_HIP)
    state['hip_R'].append(hr if hr else (0.0, 0.0))

    # --- Compute centroid for proximity checks ---
    # Use the midpoint of the bounding box approximation (shoulders + hips)
    valid_points = [p for p in [head, sl, sr, hl, hr] if p is not None]
    if valid_points:
        cx = sum(p[0] for p in valid_points) / len(valid_points)
        cy = sum(p[1] for p in valid_points) / len(valid_points)
        state['centroid'] = (cx, cy)


# ============================================================
# PROXIMITY & VIOLENCE CHECK
# ============================================================
def check_violence_between_pairs(current_time: float) -> Tuple[list, dict]:
    """
    Iterates over all pairs of tracked persons. For each pair within
    PROXIMITY_THRESHOLD distance, runs detect_advanced_violence()
    in both directions (A→B and B→A) since either could be the attacker.

    Applies a cooldown so the same pair doesn't trigger repeatedly.

    Args:
        current_time: time.time() value for cooldown comparisons.

    Returns:
        Tuple of (alerts_list, telemetry_dict)
    """
    alerts = []
    telemetry = {}
    tracked_ids = list(person_states.keys())

    # Check all unique pairs: O(n²) but n is typically < 20 people
    for i in range(len(tracked_ids)):
        for j in range(i + 1, len(tracked_ids)):
            id_a = tracked_ids[i]
            id_b = tracked_ids[j]

            state_a = person_states[id_a]
            state_b = person_states[id_b]

            # --- Proximity gate: skip distant people ---
            cx_a, cy_a = state_a['centroid']
            cx_b, cy_b = state_b['centroid']
            distance = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

            if distance > PROXIMITY_THRESHOLD:
                continue  # Too far apart — no need to analyze

            # --- Cooldown gate: skip if recently alerted ---
            pair_key = frozenset({id_a, id_b})
            last_alert = violence_cooldowns.get(pair_key, 0.0)
            if current_time - last_alert < VIOLENCE_ALERT_COOLDOWN:
                continue  # Still in cooldown window

            # --- Convert deques to lists for action_math functions ---
            history_a = {k: list(v) for k, v in state_a.items() if isinstance(v, deque)}
            history_b = {k: list(v) for k, v in state_b.items() if isinstance(v, deque)}

            # --- Check A → B (A attacks B) ---
            violence_a_to_b, telem_a_to_b = detect_advanced_violence(history_a, history_b)

            # --- Check B → A (B attacks A) ---
            violence_b_to_a, telem_b_to_a = detect_advanced_violence(history_b, history_a)

            # Record telemetry for visualization
            is_violence = violence_a_to_b or violence_b_to_a
            telemetry[pair_key] = {
                'jerk': max(telem_a_to_b['jerk'], telem_b_to_a['jerk']),
                'alignment': max(telem_a_to_b['alignment'], telem_b_to_a['alignment']),
                'is_violence': is_violence
            }

            if is_violence:
                violence_cooldowns[pair_key] = current_time
                attacker = id_a if violence_a_to_b else id_b
                victim = id_b if violence_a_to_b else id_a
                alerts.append({
                    'type': 'violence',
                    'attacker_id': int(attacker),
                    'victim_id': int(victim),
                    'pair_ids': [int(id_a), int(id_b)],
                    'timestamp': current_time,
                })
                print(f"[ALERT] Violence detected: Person {attacker} → Person {victim}")

    return alerts, telemetry


# ============================================================
# FALL CHECK
# ============================================================
def check_falls(current_time: float) -> Tuple[list, dict]:
    """
    Checks every tracked person for a fall event using the
    head velocity + torso angle heuristic.

    Args:
        current_time: time.time() value for cooldown comparisons.

    Returns:
        Tuple of (alerts_list, telemetry_dict)
    """
    alerts = []
    telemetry = {}

    for tracker_id, state in person_states.items():
        # --- Convert deques to lists ---
        history = {k: list(v) for k, v in state.items() if isinstance(v, deque)}

        width = state.get('bbox_width', 1.0)
        height = state.get('bbox_height', 1.0)
        is_fall_now, telem = detect_fall(history, width, height)
        
        # --- TEMPORAL DEBOUNCE LOGIC ---
        if is_fall_now:
            # Sudden drop + wide/crumpled detected
            state['fall_streak'] = 1
        elif state['fall_streak'] > 0:
            # They stopped dropping, but are they still on the ground?
            is_wide = telem['ar'] > 1.0
            is_crumpled = telem['crumple'] < 0.25
            if is_wide or is_crumpled:
                state['fall_streak'] += 1
            else:
                # Stood back up
                state['fall_streak'] = 0
                state['is_fallen'] = False

        # Trigger latch if they stay down for 10 frames
        if state['fall_streak'] > 10:
            state['is_fallen'] = True

        telemetry[tracker_id] = {
            'dy_norm': telem['dy_norm'],
            'ar': telem['ar'],
            'crumple': telem['crumple'],
            'is_fall': state['is_fallen']
        }

        # --- Cooldown gate ---
        last_alert = fall_cooldowns.get(tracker_id, 0.0)
        if current_time - last_alert < FALL_ALERT_COOLDOWN:
            continue

        if state['is_fallen']:
            fall_cooldowns[tracker_id] = current_time
            alerts.append({
                'type': 'fall',
                'severity': 'MEDICAL EMERGENCY',
                'person_id': int(tracker_id),
                'timestamp': current_time,
            })
            print(f"[ALERT] MEDICAL EMERGENCY (Fall detected): Person {tracker_id}")

    return alerts, telemetry


# ============================================================
# CLEANUP: Remove stale tracks
# ============================================================
def cleanup_stale_tracks(active_ids: set):
    """
    Removes person_states entries for tracker IDs that ByteTrack
    has dropped (person left frame or lost tracking). This prevents
    memory leaks during long-running surveillance sessions.

    Args:
        active_ids: Set of currently active tracker IDs from ByteTrack.
    """
    stale_ids = [tid for tid in person_states if tid not in active_ids]
    for tid in stale_ids:
        del person_states[tid]
        violence_cooldowns.pop(frozenset({tid}), None)
        fall_cooldowns.pop(tid, None)


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================
@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time video processing.

    Protocol:
      1) Client sends a video frame as base64-encoded JPEG/PNG.
      2) Server decodes, runs inference, updates state, checks alerts.
      3) Server responds with JSON containing:
         - 'detections': list of person bounding boxes + IDs
         - 'alerts': list of violence/fall alerts (if any)
         - 'frame_id': monotonic counter for client sync
    """
    await websocket.accept()
    print("[*] WebSocket client connected.")

    # Ensure models are loaded
    try:
        load_models()
    except Exception as e:
        await websocket.send_json({
            'error': f'Model loading failed: {str(e)}'
        })
        await websocket.close()
        return

    frame_count = 0

    try:
        while True:
            # --- Receive frame from client ---
            data = await websocket.receive_text()
            message = json.loads(data)

            # Decode base64 image
            img_bytes = base64.b64decode(message.get('frame', ''))
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({'error': 'Invalid frame data'})
                continue

            frame_count += 1
            current_time = time.time()

            # --- Run YOLO Pose Estimation ---
            results = pose_model(
                frame,
                conf=POSE_CONFIDENCE,
                verbose=False
            )

            detections_out = []
            active_tracker_ids = set()

            if results and results[0].keypoints is not None:
                result = results[0]

                # --- ByteTrack: update tracker with detections ---
                import supervision as sv
                sv_detections = sv.Detections.from_ultralytics(result)
                tracked = byte_tracker.update_with_detections(sv_detections)

                # --- Process each tracked person ---
                for idx in range(len(tracked)):
                    # Extract tracker ID assigned by ByteTrack
                    tracker_id = int(tracked.tracker_id[idx])
                    active_tracker_ids.add(tracker_id)

                    # Extract bounding box [x1, y1, x2, y2]
                    bbox = tracked.xyxy[idx].tolist()
                    
                    # Store bounding box dimensions for fall debounce logic
                    state = person_states[tracker_id]
                    width = max(1e-5, bbox[2] - bbox[0])
                    height = max(1e-5, bbox[3] - bbox[1])
                    state['bbox_width'] = width
                    state['bbox_height'] = height
                    state['bbox_ar'] = width / height

                    # Extract keypoints for this person
                    # result.keypoints.data shape: (num_persons, 17, 3)
                    if idx < len(result.keypoints.data):
                        kps = result.keypoints.data[idx].cpu().numpy()
                        update_person_state(tracker_id, kps)

                    detections_out.append({
                        'tracker_id': tracker_id,
                        'bbox': bbox,
                    })

            # --- Cleanup stale tracks ---
            cleanup_stale_tracks(active_tracker_ids)

            # --- Run behavior analysis ---
            violence_alerts, violence_telemetry = check_violence_between_pairs(current_time)
            fall_alerts, fall_telemetry = check_falls(current_time)
            all_alerts = violence_alerts + fall_alerts

            # --- Render Telemetry & Bounding Boxes ---
            red_boxes = set()
            for tid, t_data in fall_telemetry.items():
                if t_data['is_fall']:
                    red_boxes.add(tid)
            for pair, t_data in violence_telemetry.items():
                if t_data['is_violence']:
                    red_boxes.update(pair)

            for det in detections_out:
                tid = det['tracker_id']
                x1, y1, x2, y2 = map(int, det['bbox'])

                # Determine color: BGR format
                color = (0, 0, 255) if tid in red_boxes else (0, 255, 0)

                # Determine label text based on event detection
                label = "Normal"
                f_telem = fall_telemetry.get(tid)
                if f_telem and f_telem['is_fall']:
                    label = "MEDICAL EMERGENCY"
                else:
                    for pair, v_telem in violence_telemetry.items():
                        if tid in pair and v_telem['is_violence']:
                            label = "Violence!"
                            break

                # Draw bounding box and Status Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw Fall Telemetry
                f_telem = fall_telemetry.get(tid)
                if f_telem:
                    text = f"Drop: {f_telem['dy_norm']:.2f} | AR: {f_telem['ar']:.1f} | Crump: {f_telem['crumple']:.2f}"
                    # Display blue text below the box
                    cv2.putText(frame, text, (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw Violence Telemetry
            for pair, v_telem in violence_telemetry.items():
                ids = list(pair)
                if len(ids) == 2:
                    id_a, id_b = ids
                    if id_a in person_states and id_b in person_states:
                        cx_a, cy_a = person_states[id_a]['centroid']
                        cx_b, cy_b = person_states[id_b]['centroid']
                        mx, my = int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)
                        
                        text = f"Jerk: {v_telem['jerk']:.0f} | Align: {v_telem['alignment']:.2f}"
                        # Display purple text between the two centroids
                        cv2.putText(frame, text, (mx - 50, my), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # --- Encode modified frame to base64 ---
            _, buffer = cv2.imencode('.jpg', frame)
            frame_annotated_b64 = base64.b64encode(buffer).decode('utf-8')

            # --- Send response back to client ---
            response = {
                'frame_id': frame_count,
                'frame_annotated': frame_annotated_b64,
                'detections': detections_out,
                'alerts': all_alerts,
                'tracked_persons': len(active_tracker_ids),
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("[*] WebSocket client disconnected.")
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
        try:
            await websocket.send_json({'error': str(e)})
        except Exception:
            pass


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================
@app.get("/health")
async def health_check():
    """Simple health check for monitoring."""
    return {
        "status": "healthy",
        "tracked_persons": len(person_states),
        "models_loaded": pose_model is not None,
    }


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
