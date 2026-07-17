"""
main.py — FastAPI WebSocket Server for Real-Time Surveillance Pipeline

This is the central state manager and WebSocket endpoint. It:
  1) Receives video frames from a client over WebSocket.
  2) Runs RTMPose pose estimation + YOLOX object detection.
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
from backend.vision_models import (
    create_object_detector,
    create_pose_pipeline,
    match_keypoints_to_bbox,
)
from backend.crowd_counter import CrowdCounter
from backend.crowd_density import CrowdDensityEstimator, DensityThresholds
from backend.history_manager import PersonHistoryManager

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

# RTMPose confidence threshold for pose estimation
POSE_CONFIDENCE = 0.5

# COCO 17-point keypoint index map.
# RTMPose output is converted to this order in backend.vision_models.
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
# Central Person History Manager: bounded circular deques for bboxes, pose keypoints, joint angles, velocities, timestamps
history_manager = PersonHistoryManager(history_length=HISTORY_LENGTH, timeout_seconds=15.0)
person_states = history_manager

# Cooldown timestamps: maps frozenset({id_A, id_B}) → last_alert_time
violence_cooldowns: Dict[frozenset, float] = {}

# Cooldown timestamps: maps tracker_id → last_fall_alert_time
fall_cooldowns: Dict[int, float] = {}

# Models will be loaded lazily on first WebSocket connection
pose_model = None
detection_model = None
byte_tracker = None
crowd_counter = None
crowd_density_estimator = None


# ============================================================
# MODEL LOADING (Lazy Initialization)
# ============================================================
def load_models():
    """
    Lazily loads RTMPose/YOLOX models and ByteTrack tracker on first use.
    This avoids slow startup and allows the server to boot even
    if model files are not yet present (fails on first request
    with a clear error instead of on import).
    """
    global pose_model, detection_model, byte_tracker, crowd_counter, crowd_density_estimator

    if pose_model is None:
        try:
            import supervision as sv

            pose_model = create_pose_pipeline()
            print("[*] Pose model loaded successfully.")

            detection_model = create_object_detector()
            if detection_model is not None:
                print("[*] YOLOX custom object detector loaded successfully.")

            # ByteTrack tracker from supervision library
            byte_tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            print("[*] ByteTrack tracker initialized.")

            crowd_counter = CrowdCounter()
            crowd_density_estimator = CrowdDensityEstimator(
                thresholds=DensityThresholds(
                    medium_area_ratio=0.15,
                    high_area_ratio=0.35,
                    medium_person_count=5,
                    high_person_count=12,
                )
            )
            print("[*] Crowd analytics modules initialized.")

        except Exception as e:
            print(f"[!] Error loading models: {e}")
            raise


# Keypoint history update and derived kinematics are now handled automatically by PersonHistoryManager


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
            # Use relaxed thresholds for sustain
            is_wide = telem['ar'] > 0.7
            is_crumpled = telem['crumple'] < 0.25
            is_leaning = telem.get('torso_lean', 0.0) > 60.0
            if is_wide or is_crumpled or is_leaning:
                state['fall_streak'] += 1
            else:
                # Stood back up
                state['fall_streak'] = 0
                state['is_fallen'] = False

        # Trigger latch if they stay down for 15 frames (~0.5s @30fps)
        if state['fall_streak'] > 15:
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
def cleanup_stale_tracks(current_time: float):
    """
    Removes person_states entries that have been inactive for longer than timeout_seconds.
    This prevents memory leaks during long CCTV recordings while preserving tracks during momentary occlusions.
    """
    stale_ids = history_manager.cleanup_inactive(current_time)
    for tid in stale_ids:
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

            # --- Run person detection / initial pose estimation ---
            raw_pose = pose_model.predict(frame, confidence=POSE_CONFIDENCE)
            sv_detections = raw_pose.to_supervision()
            tracked = byte_tracker.update_with_detections(sv_detections)

            # --- Update Crowd Counting and Density Estimation ---
            if crowd_counter is not None:
                crowd_metrics = crowd_counter.update(tracked, current_time=current_time)
            else:
                crowd_metrics = None

            if crowd_density_estimator is not None:
                density_metrics = crowd_density_estimator.estimate(frame.shape, tracked)
            else:
                density_metrics = None

            detections_out = []
            active_tracker_ids = set()

            if len(tracked) > 0:
                # --- Run top-down pose estimation on tracked bounding boxes (Pre-Pose Tracking) ---
                tracked_pose = pose_model.predict(
                    frame, confidence=POSE_CONFIDENCE, tracked_boxes=tracked
                )

                # --- Process each tracked person ---
                for idx in range(len(tracked_pose)):
                    if tracked_pose.tracker_id is None:
                        continue
                    tracker_id = int(tracked_pose.tracker_id[idx])
                    active_tracker_ids.add(tracker_id)

                    # Extract bounding box [x1, y1, x2, y2]
                    bbox = tracked_pose.xyxy[idx].tolist()

                    # Update Person History Manager with new bounding box, pose keypoints, and timestamp
                    history_manager.update(tracker_id, bbox, tracked_pose.keypoints[idx], current_time)

                    detections_out.append({
                        'tracker_id': tracker_id,
                        'bbox': bbox,
                    })

            # --- Automatically remove inactive tracks after timeout ---
            cleanup_stale_tracks(current_time)

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

            # --- Draw Crowd Analytics Overlays ---
            if crowd_counter is not None and crowd_metrics is not None:
                crowd_counter.draw_hud(frame, metrics=crowd_metrics, position=(20, 40))
            if crowd_density_estimator is not None and density_metrics is not None:
                crowd_density_estimator.draw_hud(frame, metrics=density_metrics, position=(20, 145))

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
                'crowd_metrics': {
                    'active_count': crowd_metrics.current_count if crowd_metrics else 0,
                    'peak_count': crowd_metrics.peak_count if crowd_metrics else 0,
                    'cumulative_unique': crowd_metrics.cumulative_count if crowd_metrics else 0,
                },
                'crowd_density': {
                    'level': density_metrics.level.value if density_metrics else "LOW",
                    'occupied_area_percent': round(density_metrics.occupied_area_ratio * 100, 1) if density_metrics else 0.0,
                    'active_tracks': density_metrics.active_person_count if density_metrics else 0,
                },
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
