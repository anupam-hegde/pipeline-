import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# COCO-17 Keypoint Indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


def _compute_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Computes the angle (in degrees) at vertex p2 formed by vectors p2->p1 and p2->p3."""
    if p1 == (0.0, 0.0) or p2 == (0.0, 0.0) or p3 == (0.0, 0.0):
        return 0.0
    v1x, v1y = p1[0] - p2[0], p1[1] - p2[1]
    v2x, v2y = p3[0] - p2[0], p3[1] - p2[1]
    norm1 = math.hypot(v1x, v1y)
    norm2 = math.hypot(v2x, v2y)
    if norm1 < 1e-5 or norm2 < 1e-5:
        return 0.0
    dot = (v1x * v2x + v1y * v2y) / (norm1 * norm2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _extract_kp(keypoints: np.ndarray, idx: int, min_conf: float = 0.2) -> Tuple[float, float]:
    """Extracts (x, y) coordinates from COCO-17 keypoints if confidence >= min_conf."""
    if idx >= len(keypoints):
        return (0.0, 0.0)
    kp = keypoints[idx]
    if len(kp) >= 3 and kp[2] < min_conf:
        return (0.0, 0.0)
    return (float(kp[0]), float(kp[1]))


class PersonTrackHistory:
    """Encapsulates bounded temporal telemetry for a single tracked individual.

    Enforces strict O(1) memory usage via fixed-capacity deques and stores
    bounding boxes, pose keypoints, calculated joint angles, and velocities.
    """

    def __init__(self, tracker_id: int, max_length: int = 15) -> None:
        self.tracker_id = tracker_id
        self.max_length = max_length
        self.last_seen_timestamp: float = 0.0

        # Raw coordinate deques
        self.timestamps: deque[float] = deque(maxlen=max_length)
        self.bounding_boxes: deque[list[float]] = deque(maxlen=max_length)
        self.bbox_dimensions: deque[Tuple[float, float, float]] = deque(maxlen=max_length)  # (w, h, ar)
        self.pose_keypoints: deque[np.ndarray] = deque(maxlen=max_length)

        # Anatomical keypoint histories formatted for legacy action_math compatibility
        self.kp_history: Dict[str, deque[Tuple[float, float]]] = {
            'wrist_R': deque(maxlen=max_length),
            'wrist_L': deque(maxlen=max_length),
            'head': deque(maxlen=max_length),
            'shoulder_L': deque(maxlen=max_length),
            'shoulder_R': deque(maxlen=max_length),
            'hip_L': deque(maxlen=max_length),
            'hip_R': deque(maxlen=max_length),
        }

        # Computed biomechanical joint angles (in degrees)
        self.joint_angles: Dict[str, deque[float]] = {
            'elbow_R': deque(maxlen=max_length),
            'elbow_L': deque(maxlen=max_length),
            'knee_R': deque(maxlen=max_length),
            'knee_L': deque(maxlen=max_length),
            'hip_R': deque(maxlen=max_length),
            'hip_L': deque(maxlen=max_length),
        }

        # Computed 2D kinematic velocities (vx, vy) in pixels/sec
        self.velocities: Dict[str, deque[Tuple[float, float]]] = {
            'centroid': deque(maxlen=max_length),
            'wrist_R': deque(maxlen=max_length),
            'wrist_L': deque(maxlen=max_length),
            'head': deque(maxlen=max_length),
        }

        # State tracking for fall debounce heuristics
        self.centroid: Tuple[float, float] = (0.0, 0.0)
        self.fall_streak: int = 0
        self.is_fallen: bool = False

        # Medical emergency severity escalation state
        # Tracks: NONE → FALL_DETECTED → MEDICAL_EMERGENCY → CRITICAL_EMERGENCY
        self.fall_onset_time: Optional[float] = None     # timestamp when fall first confirmed
        self.immobility_start: Optional[float] = None    # timestamp when centroid stopped moving
        self.peak_severity: str = "NONE"                 # highest severity reached (never de-escalates)
        self.fall_confidence: float = 0.0                # latest weighted confidence score from detect_fall

    def update(self, bbox: list[float], keypoints: np.ndarray, timestamp: float) -> None:
        """Update historical buffers with new frame detections and compute derived kinematics."""
        prev_time = self.timestamps[-1] if self.timestamps else timestamp
        dt = max(1e-4, timestamp - prev_time) if self.timestamps else 0.0
        self.last_seen_timestamp = timestamp

        # 1. Store raw telemetry
        self.timestamps.append(timestamp)
        self.bounding_boxes.append(bbox)
        
        w = max(1e-5, float(bbox[2] - bbox[0]))
        h = max(1e-5, float(bbox[3] - bbox[1]))
        ar = w / h
        self.bbox_dimensions.append((w, h, ar))
        self.pose_keypoints.append(keypoints)

        # 2. Extract anatomical keypoints
        head = _extract_kp(keypoints, KP_NOSE)
        wr = _extract_kp(keypoints, KP_RIGHT_WRIST)
        wl = _extract_kp(keypoints, KP_LEFT_WRIST)
        sr = _extract_kp(keypoints, KP_RIGHT_SHOULDER)
        sl = _extract_kp(keypoints, KP_LEFT_SHOULDER)
        hr = _extract_kp(keypoints, KP_RIGHT_HIP)
        hl = _extract_kp(keypoints, KP_LEFT_HIP)
        er = _extract_kp(keypoints, KP_RIGHT_ELBOW)
        el = _extract_kp(keypoints, KP_LEFT_ELBOW)
        kr = _extract_kp(keypoints, KP_RIGHT_KNEE)
        kl = _extract_kp(keypoints, KP_LEFT_KNEE)
        ar_kp = _extract_kp(keypoints, KP_RIGHT_ANKLE)
        al_kp = _extract_kp(keypoints, KP_LEFT_ANKLE)

        self.kp_history['head'].append(head)
        self.kp_history['wrist_R'].append(wr)
        self.kp_history['wrist_L'].append(wl)
        self.kp_history['shoulder_R'].append(sr)
        self.kp_history['shoulder_L'].append(sl)
        self.kp_history['hip_R'].append(hr)
        self.kp_history['hip_L'].append(hl)

        # 3. Compute centroid (midpoint of shoulders + hips)
        valid_pts = [p for p in [head, sl, sr, hl, hr] if p != (0.0, 0.0)]
        prev_centroid = self.centroid
        if valid_pts:
            cx = sum(p[0] for p in valid_pts) / len(valid_pts)
            cy = sum(p[1] for p in valid_pts) / len(valid_pts)
            self.centroid = (cx, cy)
        else:
            self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        # 4. Compute Biomechanical Joint Angles
        self.joint_angles['elbow_R'].append(_compute_angle(sr, er, wr))
        self.joint_angles['elbow_L'].append(_compute_angle(sl, el, wl))
        self.joint_angles['knee_R'].append(_compute_angle(hr, kr, ar_kp))
        self.joint_angles['knee_L'].append(_compute_angle(hl, kl, al_kp))
        self.joint_angles['hip_R'].append(_compute_angle(sr, hr, kr))
        self.joint_angles['hip_L'].append(_compute_angle(sl, hl, kl))

        # 5. Compute Kinematic Velocities (pixels / second)
        if dt > 0 and len(self.timestamps) > 1:
            vx_c = (self.centroid[0] - prev_centroid[0]) / dt
            vy_c = (self.centroid[1] - prev_centroid[1]) / dt
            self.velocities['centroid'].append((vx_c, vy_c))

            prev_wr = self.kp_history['wrist_R'][-2]
            self.velocities['wrist_R'].append(((wr[0] - prev_wr[0]) / dt, (wr[1] - prev_wr[1]) / dt))

            prev_wl = self.kp_history['wrist_L'][-2]
            self.velocities['wrist_L'].append(((wl[0] - prev_wl[0]) / dt, (wl[1] - prev_wl[1]) / dt))

            prev_h = self.kp_history['head'][-2]
            self.velocities['head'].append(((head[0] - prev_h[0]) / dt, (head[1] - prev_h[1]) / dt))
        else:
            self.velocities['centroid'].append((0.0, 0.0))
            self.velocities['wrist_R'].append((0.0, 0.0))
            self.velocities['wrist_L'].append((0.0, 0.0))
            self.velocities['head'].append((0.0, 0.0))

    def get_legacy_state(self) -> Dict[str, Any]:
        """Return a dictionary matching the exact schema expected by action_math.py."""
        w, h, ar = self.bbox_dimensions[-1] if self.bbox_dimensions else (1.0, 1.0, 1.0)
        state_dict: Dict[str, Any] = {
            'wrist_R': self.kp_history['wrist_R'],
            'wrist_L': self.kp_history['wrist_L'],
            'head': self.kp_history['head'],
            'shoulder_L': self.kp_history['shoulder_L'],
            'shoulder_R': self.kp_history['shoulder_R'],
            'hip_L': self.kp_history['hip_L'],
            'hip_R': self.kp_history['hip_R'],
            'centroid': self.centroid,
            'fall_streak': self.fall_streak,
            'is_fallen': self.is_fallen,
            'bbox_ar': ar,
            'bbox_width': w,
            'bbox_height': h,
            # Medical emergency severity escalation fields
            'fall_onset_time': self.fall_onset_time,
            'immobility_start': self.immobility_start,
            'peak_severity': self.peak_severity,
            'fall_confidence': self.fall_confidence,
        }
        return state_dict

    def __getitem__(self, key: str) -> Any:
        return self.get_legacy_state()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.get_legacy_state().get(key, default)

    def items(self):
        return self.get_legacy_state().items()

    def __setitem__(self, key: str, value: Any) -> None:
        if key == 'fall_streak':
            self.fall_streak = value
        elif key == 'is_fallen':
            self.is_fallen = value
        elif key == 'centroid':
            self.centroid = value
        elif key == 'fall_onset_time':
            self.fall_onset_time = value
        elif key == 'immobility_start':
            self.immobility_start = value
        elif key == 'peak_severity':
            self.peak_severity = value
        elif key == 'fall_confidence':
            self.fall_confidence = value



class PersonHistoryManager:
    """Central repository for tracking and managing multi-frame person histories.

    Enforces configurable timeout-based eviction to prevent memory growth during
    continuous CCTV surveillance while providing clean telemetry for analytics.
    """

    def __init__(self, history_length: int = 15, timeout_seconds: float = 15.0) -> None:
        self.history_length = history_length
        self.timeout_seconds = timeout_seconds
        self._tracks: Dict[int, PersonTrackHistory] = {}

    def update(self, tracker_id: int, bbox: list[float], keypoints: np.ndarray, timestamp: float) -> PersonTrackHistory:
        """Update or initialize track history for the given ByteTrack ID."""
        if tracker_id not in self._tracks:
            self._tracks[tracker_id] = PersonTrackHistory(tracker_id, max_length=self.history_length)
        track = self._tracks[tracker_id]
        track.update(bbox, keypoints, timestamp)
        return track

    def cleanup_inactive(self, current_time: float) -> List[int]:
        """Evict tracks unseen for longer than timeout_seconds. Returns evicted IDs."""
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if (current_time - track.last_seen_timestamp) > self.timeout_seconds
        ]
        for tid in stale_ids:
            del self._tracks[tid]
        return stale_ids

    def get_history(self, tracker_id: int) -> Optional[PersonTrackHistory]:
        """Retrieve the TrackHistory instance for a specific ID."""
        return self._tracks.get(tracker_id)

    def get_legacy_state(self, tracker_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve backwards-compatible state dictionary for action_math analytics."""
        track = self._tracks.get(tracker_id)
        if track is None:
            return None
        return track.get_legacy_state()

    def items(self):
        """Iterate over (tracker_id, PersonTrackHistory) tuples."""
        return self._tracks.items()

    def keys(self):
        return self._tracks.keys()

    def values(self):
        return self._tracks.values()

    def get(self, tracker_id: int, default: Any = None) -> Any:
        return self._tracks.get(tracker_id, default)

    def __len__(self) -> int:
        return len(self._tracks)

    def __getitem__(self, tracker_id: int) -> PersonTrackHistory:
        return self._tracks[tracker_id]

    def __contains__(self, tracker_id: int) -> bool:
        return tracker_id in self._tracks
