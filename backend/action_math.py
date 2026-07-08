"""
action_math.py — Advanced Kinematics Module for Violence Detection

Implements 3rd-order kinematic analysis (Jerk) and vector alignment
(momentum transfer via dot product) to distinguish genuine violent
impacts from periodic exercise motions like jumping jacks.

Key Concepts:
  - Velocity     = 1st derivative of position (speed & direction)
  - Acceleration  = 2nd derivative of position (rate of change of velocity)
  - Jerk          = 3rd derivative of position (rate of change of acceleration)
    → A punch or strike produces an extreme jerk spike because the hand
      goes from resting → fast → contact → deceleration in very few frames.
    → Periodic exercise (jumping jacks) produces smooth sinusoidal motion
      with low jerk because there are no sudden directional changes.

  - Momentum Transfer (Dot Product Alignment):
    → After a strike lands, the victim's head recoils in roughly the same
      direction as the attacker's wrist velocity.
    → We detect this by computing the cosine similarity (dot product of
      unit vectors) between A's wrist velocity and B's head velocity.
    → A value > 0.7 indicates strong directional alignment (recoil).

YOLO Pose Keypoint Indices (COCO 17-point format):
  0: Nose        5: L-Shoulder   11: L-Hip
  1: L-Eye       6: R-Shoulder   12: R-Hip
  2: R-Eye       7: L-Elbow      13: L-Knee
  3: L-Ear       8: R-Elbow      14: R-Knee
  4: R-Ear       9: L-Wrist      15: L-Ankle
                 10: R-Wrist      16: R-Ankle
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================
# CONFIGURABLE THRESHOLDS
# ============================================================
# Minimum jerk magnitude (pixels/frame³) to classify as a violent
# impulse. This filters out smooth repetitive motions.
JERK_THRESHOLD = 100.0  # Lowered to make detection more sensitive

# Minimum cosine similarity between attacker wrist velocity and
# victim head velocity to confirm momentum transfer (recoil).
# Range: -1.0 (opposite) to 1.0 (identical direction).
MOMENTUM_TRANSFER_THRESHOLD = 0.05  # Lowered for higher sensitivity

# Minimum number of frames of history required to compute
# 3rd-order derivatives (velocity + acceleration + jerk = 3 diffs,
# so we need at least 4 points, but 5 gives a safety margin).
MIN_HISTORY_FRAMES = 5

# EMA smoothing factor. Higher alpha = less smoothing (more reactive).
# Lower alpha = more smoothing (more latency). 0.5 is a good balance
# for 30fps webcam input where YOLO jitters are ~1-3px.
DEFAULT_EMA_ALPHA = 0.5


# ============================================================
# SMOOTHING: Exponential Moving Average (EMA)
# ============================================================
def smooth_keypoints(
    history: List[Tuple[float, float]],
    alpha: float = DEFAULT_EMA_ALPHA
) -> List[Tuple[float, float]]:
    """
    Applies Exponential Moving Average (EMA) smoothing to a sequence
    of 2D keypoint coordinates to suppress YOLO pose jitter.

    EMA formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}

    Args:
        history: List of (x, y) tuples from consecutive frames.
                 history[0] is the oldest, history[-1] is the newest.
        alpha:   Smoothing coefficient in [0, 1].
                 Higher = more responsive, Lower = smoother.

    Returns:
        Smoothed list of (x, y) tuples with the same length as input.
        Returns empty list if input is empty.
    """
    if not history:
        return []

    smoothed = [history[0]]  # First point has no prior, use as-is

    for i in range(1, len(history)):
        prev_x, prev_y = smoothed[i - 1]
        curr_x, curr_y = history[i]

        # EMA: blend current observation with previous smoothed value
        sx = alpha * curr_x + (1.0 - alpha) * prev_x
        sy = alpha * curr_y + (1.0 - alpha) * prev_y
        smoothed.append((sx, sy))

    return smoothed


# ============================================================
# KINEMATICS: Velocity, Acceleration, Jerk
# ============================================================
def calculate_kinematics(
    positions: List[Tuple[float, float]],
    dt: float = 1.0
) -> Optional[Dict]:
    """
    Computes 1st through 3rd order kinematic derivatives from a
    time-series of 2D positions using NumPy vectorized operations.

    Derivatives are computed via np.diff along the time axis:
      Velocity     = diff(positions) / dt       → N-1 vectors
      Acceleration = diff(velocity) / dt        → N-2 vectors
      Jerk         = diff(acceleration) / dt    → N-3 vectors

    Args:
        positions: List of (x, y) tuples, at least 4 points.
        dt:        Time step between frames (seconds). Default 1.0
                   means units are pixels/frame, pixels/frame², etc.

    Returns:
        Dictionary with:
          'velocity_vectors':  np.array of shape (N-1, 2)
          'velocity_mag':      np.array of shape (N-1,) — scalar speeds
          'acceleration_mag':  np.array of shape (N-2,)
          'jerk_mag':          np.array of shape (N-3,)
        Returns None if fewer than 4 positions are provided.
    """
    if len(positions) < 4:
        return None

    # Convert to NumPy array: shape (N, 2)
    pos = np.array(positions, dtype=np.float64)

    # 1st derivative: Velocity vectors (direction + magnitude)
    # Shape: (N-1, 2)
    velocity = np.diff(pos, axis=0) / dt

    # Scalar speed (magnitude of each velocity vector)
    # Shape: (N-1,)
    velocity_mag = np.linalg.norm(velocity, axis=1)

    # 2nd derivative: Acceleration vectors
    # Shape: (N-2, 2)
    acceleration = np.diff(velocity, axis=0) / dt
    acceleration_mag = np.linalg.norm(acceleration, axis=1)

    # 3rd derivative: Jerk vectors
    # Shape: (N-3, 2)
    jerk = np.diff(acceleration, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=1)

    return {
        'velocity_vectors': velocity,
        'velocity_mag': velocity_mag,
        'acceleration_mag': acceleration_mag,
        'jerk_mag': jerk_mag,
    }


# ============================================================
# VIOLENCE DETECTION: Jerk + Momentum Transfer
# ============================================================
def detect_advanced_violence(
    history_A: Dict[str, List[Tuple[float, float]]],
    history_B: Dict[str, List[Tuple[float, float]]],
    jerk_threshold: float = JERK_THRESHOLD,
    momentum_threshold: float = MOMENTUM_TRANSFER_THRESHOLD,
    alpha: float = DEFAULT_EMA_ALPHA
) -> Tuple[bool, Dict[str, float]]:
    """
    Determines if Person A is striking Person B by analyzing:
      1) Jerk magnitude on A's wrist (sudden impulse detection)
      2) Dot product alignment between A's wrist velocity and B's
         head velocity (momentum transfer / recoil confirmation)

    This 2-condition AND gate eliminates false positives from:
      - Jumping jacks (high velocity but LOW jerk, periodic)
      - Waving / gesturing (moderate jerk but NO recoil on B's head)
      - Walking past someone (no jerk spike, no correlated motion)

    Args:
        history_A: Dict with keys like 'wrist_R', 'wrist_L', 'head',
                   each containing a list of (x, y) tuples from frames.
        history_B: Same structure for the potential victim.
        jerk_threshold:     Minimum jerk to flag as violent impulse.
        momentum_threshold: Minimum cosine similarity for recoil.
        alpha:              EMA smoothing factor.

    Returns:
        Tuple of (is_violence: bool, telemetry: dict).
        telemetry always contains:
          'jerk':      peak wrist jerk magnitude (pixels/frame³)
          'alignment': cosine similarity of wrist→head velocity ([-1, 1])
        These raw values are used by main.py to render the telemetry overlay.
    """
    # Zero telemetry sentinel — returned on any early exit
    _null_telem: Dict[str, float] = {'jerk': 0.0, 'alignment': 0.0}

    # Victim: use head/nose (index 0)
    head_history_B = history_B.get('head', [])
    if len(head_history_B) < MIN_HISTORY_FRAMES:
        return False, _null_telem

    head_history_B = _fill_missing(head_history_B)
    smoothed_head_B = smooth_keypoints(head_history_B, alpha)
    kin_head_B = calculate_kinematics(smoothed_head_B)
    if kin_head_B is None:
        return False, _null_telem

    head_vel = kin_head_B['velocity_vectors'][-1]

    best_jerk = 0.0
    best_cos_sim = 0.0
    is_violence = False

    # Check both wrists for the attacker
    for wrist_key in ['wrist_R', 'wrist_L']:
        wrist_history_A = history_A.get(wrist_key, [])
        if len(wrist_history_A) < MIN_HISTORY_FRAMES:
            continue

        wrist_history_A = _fill_missing(wrist_history_A)
        smoothed_wrist_A = smooth_keypoints(wrist_history_A, alpha)
        kin_wrist_A = calculate_kinematics(smoothed_wrist_A)
        
        if kin_wrist_A is None:
            continue

        recent_jerk = kin_wrist_A['jerk_mag'][-3:]
        peak_jerk = float(np.max(recent_jerk)) if len(recent_jerk) > 0 else 0.0
        wrist_vel = kin_wrist_A['velocity_vectors'][-1]

        cos_sim = _cosine_similarity(wrist_vel, head_vel)

        if peak_jerk > best_jerk:
            best_jerk = peak_jerk
            best_cos_sim = cos_sim

        # Condition 1: Extreme jerk (wild punch or push) - triggers even with low/no recoil
        if peak_jerk > jerk_threshold * 1.5 and cos_sim > 0.0:
            is_violence = True
            
        # Condition 2: High jerk AND recoil alignment
        elif peak_jerk >= jerk_threshold and cos_sim >= momentum_threshold:
            is_violence = True

    telemetry: Dict[str, float] = {
        'jerk': best_jerk,
        'alignment': best_cos_sim,
    }

    return is_violence, telemetry


# ============================================================
# FALL DETECTION (Robust CV Heuristic)
# ============================================================
def detect_fall(
    history: Dict[str, List[Tuple[float, float]]],
    bbox_width: float,
    bbox_height: float,
    velocity_threshold_norm: float = 0.05,  # 5% of body height per frame drop
    crumple_threshold_norm: float = 0.25    # Head is within 25% of body height from hips
) -> Tuple[bool, Dict[str, float]]:
    """
    Detects a fall or medical emergency based on scale-invariant 
    multi-factor heuristics.

    Args:
        history:  Dict with keypoint histories.
        bbox_width: Current bounding box width.
        bbox_height: Current bounding box height.
        velocity_threshold_norm: Min downward speed (normalized by bbox height).
        crumple_threshold_norm: Max head-to-hip vertical distance (normalized) to be considered crumpled.

    Returns:
        Tuple of (is_fall_now: bool, telemetry: dict).
        telemetry always contains:
          'dy_norm': downward head velocity normalized by body height
          'ar': bounding box aspect ratio
          'crumple': head-to-hip distance normalized by body height
    """
    _null_telem: Dict[str, float] = {'dy_norm': 0.0, 'ar': 0.0, 'crumple': 0.0}

    if bbox_height < 1e-5:
        return False, _null_telem

    head_hist = history.get('head', [])
    if len(head_hist) < 3:
        return False, _null_telem

    # --- 1. Scale-Invariant Velocity ---
    smoothed_head = smooth_keypoints(head_hist)
    recent_positions = smoothed_head[-3:]
    dy = float(recent_positions[-1][1] - recent_positions[-2][1])
    dy_norm = dy / bbox_height

    # --- 2. Aspect Ratio (Floor Test) ---
    bbox_ar = bbox_width / bbox_height

    # --- 3. Crumple Detection (Slump Test) ---
    hip_L = history.get('hip_L', [])
    hip_R = history.get('hip_R', [])
    
    crumple_ratio = 1.0
    if hip_L and hip_R:
        hip_mid_y = (hip_L[-1][1] + hip_R[-1][1]) / 2.0
        head_y = recent_positions[-1][1]
        
        # In image coords, Y increases downwards, so hips are usually > head
        # Distance from head down to hips
        head_to_hip_dist = max(0.0, hip_mid_y - head_y)
        crumple_ratio = head_to_hip_dist / bbox_height

    # Build telemetry
    telemetry: Dict[str, float] = {
        'dy_norm': dy_norm,
        'ar': bbox_ar,
        'crumple': crumple_ratio
    }

    # Condition 1: Fast scale-invariant drop
    is_dropping = dy_norm > velocity_threshold_norm
    
    # Condition 2: Wide aspect ratio (on the ground)
    is_wide = bbox_ar > 1.0
    
    # Condition 3: Crumpled (head very close to hips vertically)
    is_crumpled = crumple_ratio < crumple_threshold_norm

    # Fall triggers if dropping AND (wide OR crumpled)
    if is_dropping and (is_wide or is_crumpled):
        return True, telemetry
        
    return False, telemetry


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def _fill_missing(
    history: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Replaces (0, 0) entries (undetected keypoints) with the last
    known valid position. This prevents massive artificial jerk
    spikes when a keypoint briefly disappears and reappears.

    If the very first entry is (0, 0), it is left as-is (we have
    no prior reference). Subsequent (0, 0) values are forward-filled.

    Args:
        history: List of (x, y) keypoint positions.

    Returns:
        Cleaned list with (0, 0) values replaced.
    """
    if not history:
        return history

    cleaned = [history[0]]
    for i in range(1, len(history)):
        x, y = history[i]
        if x == 0.0 and y == 0.0:
            # Keypoint was not detected — carry forward previous value
            cleaned.append(cleaned[i - 1])
        else:
            cleaned.append((x, y))

    return cleaned


def _cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray
) -> float:
    """
    Computes the cosine similarity between two 2D vectors.

    Returns a value in [-1, 1]:
      +1.0 = identical direction
       0.0 = perpendicular
      -1.0 = opposite direction

    Returns 0.0 if either vector has zero magnitude (prevents NaN).

    Args:
        vec_a: 2D vector as numpy array.
        vec_b: 2D vector as numpy array.

    Returns:
        Cosine similarity as a float.
    """
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)

    # Guard against division by zero (stationary keypoint)
    if mag_a < 1e-8 or mag_b < 1e-8:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (mag_a * mag_b))


def extract_keypoint(
    keypoints: np.ndarray,
    index: int
) -> Optional[Tuple[float, float]]:
    """
    Safely extracts a single (x, y) keypoint from a YOLO pose result.

    YOLO pose keypoints come as a (17, 3) array where each row is
    [x, y, confidence]. This function checks bounds, validates
    confidence, and returns None for low-confidence / missing points.

    Args:
        keypoints: NumPy array of shape (17, 2) or (17, 3).
        index:     COCO keypoint index (0-16).

    Returns:
        (x, y) tuple if valid, None if missing or low confidence.
    """
    if keypoints is None or len(keypoints) == 0:
        return None

    if index < 0 or index >= len(keypoints):
        return None

    kp = keypoints[index]

    # Handle (17, 3) format with confidence score
    if len(kp) >= 3:
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf < 0.3:  # Low confidence — treat as undetected
            return None
        return (x, y)

    # Handle (17, 2) format without confidence
    x, y = float(kp[0]), float(kp[1])
    if x == 0.0 and y == 0.0:
        return None

    return (x, y)
