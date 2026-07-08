import os
import sys
import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

# Ensure backend package is in path so we can import action_math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.action_math import (
    detect_advanced_violence,
    detect_fall,
    extract_keypoint,
)
from backend.vision_models import (
    ObjectDetectionBatch,
    create_object_detector,
    create_pose_pipeline,
    draw_object_detections,
    match_keypoints_to_bbox,
)

# Constants
HISTORY_LENGTH = 15
PROXIMITY_THRESHOLD = 500.0  # Increased to handle subjects close to camera

# COCO 17-point format indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10


def update_person_state(tracker_id: int, keypoints: np.ndarray, state: dict):
    """
    Extracts relevant keypoints from a YOLO pose result and appends
    them to the rolling history deque for the given tracked person.
    """
    # Extract
    head = extract_keypoint(keypoints, KP_NOSE)
    wr = extract_keypoint(keypoints, KP_RIGHT_WRIST)
    wl = extract_keypoint(keypoints, KP_LEFT_WRIST)
    sl = extract_keypoint(keypoints, KP_LEFT_SHOULDER)
    sr = extract_keypoint(keypoints, KP_RIGHT_SHOULDER)
    hl = extract_keypoint(keypoints, KP_LEFT_HIP)
    hr = extract_keypoint(keypoints, KP_RIGHT_HIP)

    # Append (use (0,0) if missing)
    state['head'].append(head if head else (0.0, 0.0))
    state['wrist_R'].append(wr if wr else (0.0, 0.0))
    state['wrist_L'].append(wl if wl else (0.0, 0.0))
    state['shoulder_L'].append(sl if sl else (0.0, 0.0))
    state['shoulder_R'].append(sr if sr else (0.0, 0.0))
    state['hip_L'].append(hl if hl else (0.0, 0.0))
    state['hip_R'].append(hr if hr else (0.0, 0.0))

    # Centroid
    valid_points = [p for p in [head, sl, sr, hl, hr] if p is not None]
    if valid_points:
        cx = sum(p[0] for p in valid_points) / len(valid_points)
        cy = sum(p[1] for p in valid_points) / len(valid_points)
        state['centroid'] = (cx, cy)


def main():
    parser = argparse.ArgumentParser(description="Offline Video Test for Surveillance Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video")
    args = parser.parse_args()

    # 1. Setup and Imports (Models & Tracker)
    print("[*] Loading models...")
    print("[*] Loading RTMPose model...")
    pose_model = create_pose_pipeline()
    
    print("[*] Loading optional YOLOX custom object detector...")
    det_model = create_object_detector()

    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    # 2. Video I/O Handling
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[!] Error opening video file: {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 3. Core Processing Loop
    person_states = defaultdict(lambda: {
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
    })

    frame_idx = 0
    print(f"[*] Starting processing loop for {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # a. Run RTMPose pose estimation
        pose_results = pose_model.predict(frame, confidence=0.5)
        
        # Run custom YOLOX object detection for fire and weapons when configured.
        det_results = (
            det_model.predict(frame, confidence=0.45)
            if det_model is not None
            else ObjectDetectionBatch.empty()
        )
        
        detections = pose_results.to_supervision()

        # b. Pass to ByteTrack
        tracked = byte_tracker.update_with_detections(detections)

        active_tracker_ids = set()
        
        red_boxes = set()
        fall_telemetry = {}
        violence_telemetry = {}

        # c. Update history for tracked detections
        for idx in range(len(tracked)):
            tracker_id = int(tracked.tracker_id[idx])
            active_tracker_ids.add(tracker_id)
            
            # Match bounding box to find the correct keypoints
            tracked_bbox = tracked.xyxy[idx]
            
            # Calculate and store bounding box Aspect Ratio for fall reset logic
            width = max(1e-5, tracked_bbox[2] - tracked_bbox[0])
            height = max(1e-5, tracked_bbox[3] - tracked_bbox[1])
            person_states[tracker_id]['bbox_ar'] = width / height
            
            kps = match_keypoints_to_bbox(pose_results, tracked_bbox)
            if kps is not None:
                update_person_state(tracker_id, kps, person_states[tracker_id])

            # d. Run detect_fall() and capture telemetry
            history = {k: list(v) for k, v in person_states[tracker_id].items() if isinstance(v, deque)}
            is_fall_now, t_fall = detect_fall(history, width, height)
            
            # --- TEMPORAL DEBOUNCE LOGIC ---
            if is_fall_now:
                # Sudden drop + wide/crumpled detected
                person_states[tracker_id]['fall_streak'] = 1
            elif person_states[tracker_id]['fall_streak'] > 0:
                # They stopped dropping, but are they still on the ground?
                is_wide = t_fall['ar'] > 1.0
                is_crumpled = t_fall['crumple'] < 0.25
                if is_wide or is_crumpled:
                    person_states[tracker_id]['fall_streak'] += 1
                else:
                    # Stood back up (e.g., tying shoe)
                    person_states[tracker_id]['fall_streak'] = 0
                    person_states[tracker_id]['is_fallen'] = False

            # Trigger latch if they stay down for 10 frames
            if person_states[tracker_id]['fall_streak'] > 10:
                person_states[tracker_id]['is_fallen'] = True
            
            fall_telemetry[tracker_id] = {
                'dy_norm': t_fall['dy_norm'],
                'ar': t_fall['ar'],
                'crumple': t_fall['crumple'],
                'is_fall': person_states[tracker_id]['is_fallen']
            }
            if person_states[tracker_id]['is_fallen']:
                red_boxes.add(tracker_id)

        # Cleanup stale tracks
        stale_ids = [tid for tid in person_states if tid not in active_tracker_ids]
        for tid in stale_ids:
            del person_states[tid]

        # e. Calculate proximity & run advanced violence
        tracked_ids = list(person_states.keys())
        for i in range(len(tracked_ids)):
            for j in range(i + 1, len(tracked_ids)):
                id_a = tracked_ids[i]
                id_b = tracked_ids[j]

                cx_a, cy_a = person_states[id_a]['centroid']
                cx_b, cy_b = person_states[id_b]['centroid']
                dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)

                if dist < PROXIMITY_THRESHOLD:
                    hist_a = {k: list(v) for k, v in person_states[id_a].items() if isinstance(v, deque)}
                    hist_b = {k: list(v) for k, v in person_states[id_b].items() if isinstance(v, deque)}

                    v_a_b, telem_a_b = detect_advanced_violence(hist_a, hist_b)
                    v_b_a, telem_b_a = detect_advanced_violence(hist_b, hist_a)

                    is_violence = v_a_b or v_b_a
                    if is_violence:
                        red_boxes.update([id_a, id_b])

                    violence_telemetry[(id_a, id_b)] = {
                        'jerk': max(telem_a_b['jerk'], telem_b_a['jerk']),
                        'alignment': max(telem_a_b['alignment'], telem_b_a['alignment']),
                        'is_violence': is_violence
                    }

        # 4. Visual Annotation Overlay
        headcount = len(tracked)
        cv2.putText(frame, f"Headcount: {headcount}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        for idx in range(len(tracked)):
            tid = int(tracked.tracker_id[idx])
            x1, y1, x2, y2 = map(int, tracked.xyxy[idx])
            
            # Color: Red if fall/violence, else Green
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

            # Draw Bounding Box & Status Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Burn Fall Telemetry
            f_telem = fall_telemetry.get(tid)
            if f_telem:
                text = f"Drop: {f_telem['dy_norm']:.2f} | AR: {f_telem['ar']:.1f} | Crump: {f_telem['crumple']:.2f}"
                cv2.putText(frame, text, (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
        # Burn Violence Telemetry
        for pair, v_telem in violence_telemetry.items():
            id_a, id_b = pair
            if id_a in person_states and id_b in person_states:
                cx_a, cy_a = person_states[id_a]['centroid']
                cx_b, cy_b = person_states[id_b]['centroid']
                mx, my = int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)
                
                text = f"Jerk: {v_telem['jerk']:.0f} | Align: {v_telem['alignment']:.2f}"
                cv2.putText(frame, text, (mx - 50, my), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        draw_object_detections(frame, det_results)

        # Write frame to output video
        out.write(frame)

        # 5. Live Preview
        cv2.imshow('Offline Testing Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Loop broken by user.")
            break
            
        if frame_idx % 30 == 0:
            print(f"[*] Processed {frame_idx}/{total_frames} frames...")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[*] Processing complete. Saved to {args.output}")

if __name__ == "__main__":
    main()
