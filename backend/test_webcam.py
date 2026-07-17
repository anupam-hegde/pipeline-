import os
import sys
import time
import argparse
from collections import deque

import cv2
import numpy as np
import supervision as sv

# Ensure backend package is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.action_math import (
    detect_advanced_violence,
    detect_fall,
)
from backend.vision_models import (
    ObjectDetectionBatch,
    create_object_detector,
    create_pose_pipeline,
    draw_object_detections,
)
from backend.crowd_counter import CrowdCounter
from backend.crowd_density import CrowdDensityEstimator, DensityThresholds
from backend.history_manager import PersonHistoryManager

# Constants
HISTORY_LENGTH = 15
PROXIMITY_THRESHOLD = 500.0  # Increased to handle subjects close to camera


def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Test for Surveillance Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal mirroring")
    args = parser.parse_args()

    print("[*] Loading RTMPose model...")
    pose_model = create_pose_pipeline()
    
    print("[*] Loading optional YOLOX custom object detector...")
    det_model = create_object_detector()
    
    print("[*] Initializing ByteTrack...")
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    print("[*] Initializing Crowd Analytics & History Manager...")
    crowd_counter = CrowdCounter()
    crowd_density_estimator = CrowdDensityEstimator(
        thresholds=DensityThresholds(
            medium_area_ratio=0.15,
            high_area_ratio=0.35,
            medium_person_count=5,
            high_person_count=12,
        )
    )
    history_manager = PersonHistoryManager(history_length=HISTORY_LENGTH, timeout_seconds=15.0)

    print(f"[*] Opening Camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Error opening camera index {args.camera}")
        return

    # Set camera resolutions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("[*] Starting Live Camera Feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from camera.")
            break
            
        # Flip frame horizontally for intuitive mirror effect unless disabled
        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        current_time = time.time()

        # Run RTMPose pose estimation for initial detections
        raw_pose = pose_model.predict(frame, confidence=0.5)
        
        # Run custom YOLOX object detection for fire and weapons when configured
        det_results = (
            det_model.predict(frame, confidence=0.45)
            if det_model is not None
            else ObjectDetectionBatch.empty()
        )
        
        detections = raw_pose.to_supervision()

        # Pass to ByteTrack
        tracked = byte_tracker.update_with_detections(detections)

        # Update Crowd Counting and Density Estimation
        crowd_metrics = crowd_counter.update(tracked, current_time=current_time)
        density_metrics = crowd_density_estimator.estimate(frame.shape, tracked)

        active_tracker_ids = set()
        red_boxes = set()
        fall_telemetry = {}
        violence_telemetry = {}

        if len(tracked) > 0:
            # Run top-down pose estimation on tracked bounding boxes
            tracked_pose = pose_model.predict(
                frame, confidence=0.5, tracked_boxes=tracked
            )

            for idx in range(len(tracked_pose)):
                if tracked_pose.tracker_id is None:
                    continue
                tracker_id = int(tracked_pose.tracker_id[idx])
                active_tracker_ids.add(tracker_id)

                bbox = tracked_pose.xyxy[idx].tolist()
                kps = tracked_pose.keypoints[idx]

                # Update PersonHistoryManager
                track_state = history_manager.update(tracker_id, bbox, kps, current_time)

                # Detect fall
                history = {k: list(v) for k, v in track_state.items() if isinstance(v, deque)}
                is_fall_now, t_fall = detect_fall(
                    history,
                    track_state.get('bbox_width', 1.0),
                    track_state.get('bbox_height', 1.0),
                )

                # Temporal Debounce Logic
                if is_fall_now:
                    track_state['fall_streak'] = 1
                elif track_state['fall_streak'] > 0:
                    # Use relaxed thresholds for sustain
                    is_wide = t_fall['ar'] > 0.7
                    is_crumpled = t_fall['crumple'] < 0.25
                    is_leaning = t_fall.get('torso_lean', 0.0) > 60.0
                    if is_wide or is_crumpled or is_leaning:
                        track_state['fall_streak'] += 1
                    else:
                        track_state['fall_streak'] = 0
                        track_state['is_fallen'] = False

                if track_state['fall_streak'] > 15:
                    track_state['is_fallen'] = True

                fall_telemetry[tracker_id] = {
                    'dy_norm': t_fall['dy_norm'],
                    'ar': t_fall['ar'],
                    'crumple': t_fall['crumple'],
                    'is_fall': track_state['is_fallen'],
                }
                if track_state['is_fallen']:
                    red_boxes.add(tracker_id)

        # Cleanup stale tracks
        history_manager.cleanup_inactive(current_time)

        # Calculate proximity & run advanced violence
        tracked_ids = list(history_manager.keys())
        for i in range(len(tracked_ids)):
            for j in range(i + 1, len(tracked_ids)):
                id_a = tracked_ids[i]
                id_b = tracked_ids[j]

                state_a = history_manager[id_a]
                state_b = history_manager[id_b]

                cx_a, cy_a = state_a['centroid']
                cx_b, cy_b = state_b['centroid']
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

                if dist < PROXIMITY_THRESHOLD:
                    hist_a = {k: list(v) for k, v in state_a.items() if isinstance(v, deque)}
                    hist_b = {k: list(v) for k, v in state_b.items() if isinstance(v, deque)}

                    v_a_b, telem_a_b = detect_advanced_violence(hist_a, hist_b)
                    v_b_a, telem_b_a = detect_advanced_violence(hist_b, hist_a)

                    is_violence = v_a_b or v_b_a
                    if is_violence:
                        red_boxes.update([id_a, id_b])

                    violence_telemetry[(id_a, id_b)] = {
                        'jerk': max(telem_a_b['jerk'], telem_b_a['jerk']),
                        'alignment': max(telem_a_b['alignment'], telem_b_a['alignment']),
                        'is_violence': is_violence,
                    }

        # Draw Person Bounding Boxes & Alerts
        if len(tracked) > 0 and tracked_pose.xyxy is not None:
            for idx in range(len(tracked_pose)):
                if tracked_pose.tracker_id is None:
                    continue
                tid = int(tracked_pose.tracker_id[idx])
                x1, y1, x2, y2 = map(int, tracked_pose.xyxy[idx])

                color = (0, 0, 255) if tid in red_boxes else (0, 255, 0)

                label = "Normal"
                f_telem = fall_telemetry.get(tid)
                if f_telem and f_telem['is_fall']:
                    label = "MEDICAL EMERGENCY"
                else:
                    for pair, v_telem in violence_telemetry.items():
                        if tid in pair and v_telem['is_violence']:
                            label = "Violence!"
                            break

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{tid} {label}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if f_telem:
                    text = f"Drop: {f_telem['dy_norm']:.2f} | AR: {f_telem['ar']:.1f} | Crump: {f_telem['crumple']:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

        # Burn Violence Telemetry
        for pair, v_telem in violence_telemetry.items():
            id_a, id_b = pair
            if id_a in history_manager and id_b in history_manager:
                cx_a, cy_a = history_manager[id_a]['centroid']
                cx_b, cy_b = history_manager[id_b]['centroid']
                mx, my = int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)

                text = f"Jerk: {v_telem['jerk']:.0f} | Align: {v_telem['alignment']:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (mx - 50, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

        draw_object_detections(frame, det_results)

        # Draw Crowd Analytics HUD Overlays
        crowd_counter.draw_hud(frame, metrics=crowd_metrics, position=(20, 40))
        crowd_density_estimator.draw_hud(frame, metrics=density_metrics, position=(20, 145))

        # Live Preview
        cv2.imshow('Real-Time Surveillance Pipeline - Live Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Loop broken by user.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Camera feed closed.")


if __name__ == "__main__":
    main()

