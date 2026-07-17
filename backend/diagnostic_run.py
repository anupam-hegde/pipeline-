"""
Diagnostic script: dumps all per-frame telemetry to CSV for threshold calibration.
Processes test video and logs jerk, wrist speed, head speed, cos_sim, fall metrics
for every tracked person pair so we can see what real violence looks like numerically.
"""
import os, sys, csv, time
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.action_math import (
    smooth_keypoints, calculate_kinematics, _fill_missing, _cosine_similarity,
    detect_fall, extract_keypoint, MIN_HISTORY_FRAMES, DEFAULT_EMA_ALPHA,
)
from backend.vision_models import (
    ObjectDetectionBatch, create_object_detector, create_pose_pipeline,
    match_keypoints_to_bbox,
)

HISTORY_LENGTH = 15
KP_NOSE, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 0, 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_WRIST, KP_RIGHT_WRIST = 11, 12, 9, 10


def update_person_state(tracker_id, keypoints, state):
    from backend.action_math import extract_keypoint
    head = extract_keypoint(keypoints, KP_NOSE)
    wr = extract_keypoint(keypoints, KP_RIGHT_WRIST)
    wl = extract_keypoint(keypoints, KP_LEFT_WRIST)
    sl = extract_keypoint(keypoints, KP_LEFT_SHOULDER)
    sr = extract_keypoint(keypoints, KP_RIGHT_SHOULDER)
    hl = extract_keypoint(keypoints, KP_LEFT_HIP)
    hr = extract_keypoint(keypoints, KP_RIGHT_HIP)

    state['head'].append(head if head else (0.0, 0.0))
    state['wrist_R'].append(wr if wr else (0.0, 0.0))
    state['wrist_L'].append(wl if wl else (0.0, 0.0))
    state['shoulder_L'].append(sl if sl else (0.0, 0.0))
    state['shoulder_R'].append(sr if sr else (0.0, 0.0))
    state['hip_L'].append(hl if hl else (0.0, 0.0))
    state['hip_R'].append(hr if hr else (0.0, 0.0))

    valid_points = [p for p in [head, sl, sr, hl, hr] if p is not None]
    if valid_points:
        cx = sum(p[0] for p in valid_points) / len(valid_points)
        cy = sum(p[1] for p in valid_points) / len(valid_points)
        state['centroid'] = (cx, cy)


def compute_violence_signals(history_A, history_B, alpha=DEFAULT_EMA_ALPHA):
    """Compute raw signal values WITHOUT thresholding — for diagnostic logging."""
    results = {'peak_jerk': 0.0, 'wrist_speed': 0.0, 'head_speed': 0.0, 'cos_sim': 0.0}

    head_history_B = history_B.get('head', [])
    if len(head_history_B) < MIN_HISTORY_FRAMES:
        return results
    head_history_B = _fill_missing(head_history_B)
    smoothed_head_B = smooth_keypoints(head_history_B, alpha)
    kin_head_B = calculate_kinematics(smoothed_head_B)
    if kin_head_B is None:
        return results

    head_vel = kin_head_B['velocity_vectors'][-1]
    head_speed = float(np.linalg.norm(head_vel))
    results['head_speed'] = head_speed

    for wrist_key in ['wrist_R', 'wrist_L']:
        wrist_history_A = history_A.get(wrist_key, [])
        if len(wrist_history_A) < MIN_HISTORY_FRAMES:
            continue
        wrist_history_A = _fill_missing(wrist_history_A)
        smoothed = smooth_keypoints(wrist_history_A, alpha)
        kin = calculate_kinematics(smoothed)
        if kin is None:
            continue

        recent_jerk = kin['jerk_mag'][-5:]
        peak_jerk = float(np.max(recent_jerk)) if len(recent_jerk) > 0 else 0.0
        wrist_vel = kin['velocity_vectors'][-1]
        wrist_speed = float(np.linalg.norm(wrist_vel))
        cos_sim = _cosine_similarity(wrist_vel, head_vel)

        if peak_jerk > results['peak_jerk']:
            results['peak_jerk'] = peak_jerk
            results['wrist_speed'] = wrist_speed
            results['cos_sim'] = cos_sim

    return results


def main():
    input_path = "test_video (2).mp4"
    csv_path = "diagnostic_telemetry.csv"

    print("[*] Loading models...")
    pose_model = create_pose_pipeline()
    det_model = create_object_detector()
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25, lost_track_buffer=30,
        minimum_matching_threshold=0.8, frame_rate=30,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] Error opening {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    person_states = defaultdict(lambda: {
        'wrist_R': deque(maxlen=HISTORY_LENGTH), 'wrist_L': deque(maxlen=HISTORY_LENGTH),
        'head': deque(maxlen=HISTORY_LENGTH),
        'shoulder_L': deque(maxlen=HISTORY_LENGTH), 'shoulder_R': deque(maxlen=HISTORY_LENGTH),
        'hip_L': deque(maxlen=HISTORY_LENGTH), 'hip_R': deque(maxlen=HISTORY_LENGTH),
        'centroid': (0.0, 0.0), 'fall_streak': 0, 'is_fallen': False, 'bbox_ar': 1.0,
    })

    # CSV columns
    violence_rows = []
    fall_rows = []

    frame_idx = 0
    print(f"[*] Processing {total_frames} frames for diagnostics...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        pose_results = pose_model.predict(frame, confidence=0.5)
        detections = pose_results.to_supervision()
        tracked = byte_tracker.update_with_detections(detections)

        active_ids = set()
        for idx in range(len(tracked)):
            tid = int(tracked.tracker_id[idx])
            active_ids.add(tid)
            bbox = tracked.xyxy[idx]
            w = max(1e-5, bbox[2] - bbox[0])
            h = max(1e-5, bbox[3] - bbox[1])
            person_states[tid]['bbox_ar'] = w / h

            kps = match_keypoints_to_bbox(pose_results, bbox)
            if kps is not None:
                update_person_state(tid, kps, person_states[tid])

            # Fall telemetry
            history = {k: list(v) for k, v in person_states[tid].items() if isinstance(v, deque)}
            _, t_fall = detect_fall(history, w, h)
            fall_rows.append({
                'frame': frame_idx, 'tracker_id': tid,
                'dy_norm': round(t_fall['dy_norm'], 4), 'ar': round(t_fall['ar'], 3),
                'crumple': round(t_fall['crumple'], 4),
                'torso_lean': round(t_fall.get('torso_lean', 0.0), 2),
                'bbox_w': round(float(w), 1), 'bbox_h': round(float(h), 1),
            })

        # Cleanup stale
        stale = [t for t in person_states if t not in active_ids]
        for t in stale:
            del person_states[t]

        # Violence pairs
        tracked_ids = list(person_states.keys())
        for i in range(len(tracked_ids)):
            for j in range(i + 1, len(tracked_ids)):
                id_a, id_b = tracked_ids[i], tracked_ids[j]
                cx_a, cy_a = person_states[id_a]['centroid']
                cx_b, cy_b = person_states[id_b]['centroid']
                dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)

                if dist < 500:
                    hist_a = {k: list(v) for k, v in person_states[id_a].items() if isinstance(v, deque)}
                    hist_b = {k: list(v) for k, v in person_states[id_b].items() if isinstance(v, deque)}

                    sig_ab = compute_violence_signals(hist_a, hist_b)
                    sig_ba = compute_violence_signals(hist_b, hist_a)

                    violence_rows.append({
                        'frame': frame_idx, 'id_a': id_a, 'id_b': id_b,
                        'dist': round(dist, 1),
                        'jerk_ab': round(sig_ab['peak_jerk'], 1),
                        'wrist_spd_ab': round(sig_ab['wrist_speed'], 1),
                        'head_spd_ab': round(sig_ab['head_speed'], 1),
                        'cos_sim_ab': round(sig_ab['cos_sim'], 3),
                        'jerk_ba': round(sig_ba['peak_jerk'], 1),
                        'wrist_spd_ba': round(sig_ba['wrist_speed'], 1),
                        'head_spd_ba': round(sig_ba['head_speed'], 1),
                        'cos_sim_ba': round(sig_ba['cos_sim'], 3),
                    })

        if frame_idx % 30 == 0:
            print(f"  [{frame_idx}/{total_frames}]")

    cap.release()

    # Write CSVs
    with open("diag_violence.csv", 'w', newline='') as f:
        if violence_rows:
            w = csv.DictWriter(f, fieldnames=violence_rows[0].keys())
            w.writeheader()
            w.writerows(violence_rows)
    print(f"[*] Violence telemetry: diag_violence.csv ({len(violence_rows)} rows)")

    with open("diag_fall.csv", 'w', newline='') as f:
        if fall_rows:
            w = csv.DictWriter(f, fieldnames=fall_rows[0].keys())
            w.writeheader()
            w.writerows(fall_rows)
    print(f"[*] Fall telemetry: diag_fall.csv ({len(fall_rows)} rows)")
    print("[*] Done! Analyze CSVs to find optimal thresholds.")


if __name__ == "__main__":
    main()
