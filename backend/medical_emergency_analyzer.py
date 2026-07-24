"""
medical_emergency_analyzer.py — Rule-Based Medical Emergency Fall Detection Module

This production-grade module implements 10 strictly rule-based biomechanical and
kinematic checks using ByteTrack Track IDs and COCO-17 RTMPose keypoints.

Rules Implemented:
  1. Rapid vertical body descent.
  2. Torso changes from vertical to horizontal.
  3. Bounding box aspect ratio changes from standing to lying.
  4. Hip and shoulder alignment indicates lying posture.
  5. Head moves close to the floor.
  6. Body center suddenly drops.
  7. Large movement followed by an abrupt stop.
  8. Very low body movement for 5–10 seconds after the fall.
  9. Detect whether the person recovers (stands up). If recovery occurs, cancel the emergency.
  10. Trigger a Medical Emergency only if the person remains lying and nearly motionless after the confirmation period.

Confidence Scoring Levels:
  - 0–39: Normal
  - 40–69: Possible Fall
  - 70–89: High Probability Fall
  - 90–100: Medical Emergency
"""

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

# Configure module-level logging
logger = logging.getLogger("MedicalEmergencyAnalyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# COCO-17 Keypoint Indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


@dataclass
class MedicalEmergencyConfig:
    """Configurable thresholds and weights for all 10 biomechanical fall rules."""
    
    # --- Biomechanical Thresholds ---
    rule1_rapid_descent_velocity_norm: float = 1.3       # Box heights/sec downward speed
    rule2_torso_horizontal_angle_deg: float = 62.0       # Angle from vertical (>= 62 deg is horizontal)
    rule2_torso_upright_angle_deg: float = 38.0          # Angle from vertical (<= 38 deg is upright)
    rule3_aspect_ratio_lying: float = 1.05               # Width / Height >= 1.05 indicates lying
    rule3_aspect_ratio_standing: float = 0.68            # Width / Height <= 0.68 indicates upright
    rule4_hip_shoulder_alignment_norm: float = 0.28      # |y_shoulder - y_hip| / h <= 0.28 indicates flat
    rule5_head_floor_proximity_norm: float = 0.22        # (y_bottom - y_nose) / h <= 0.22 indicates head on floor
    rule6_center_drop_norm: float = 0.32                 # Net vertical drop in centroid / h over 1.5s window
    rule7_high_velocity_threshold_norm: float = 1.0      # Peak velocity prior to impact
    rule7_abrupt_stop_velocity_norm: float = 0.05        # Post-impact velocity floor
    rule8_immobility_speed_px_sec: float = 8.0           # Max average speed during 5-10s post-fall window
    rule9_recovery_torso_angle_deg: float = 40.0         # Torso angle return indicating standing up
    rule9_recovery_aspect_ratio: float = 0.70            # Aspect ratio return indicating standing up
    rule10_confirmation_period_sec: float = 5.0          # Required continuous lying + motionless duration for Medical Emergency

    # --- Rule Weights (Suming to 100 base points across indicators) ---
    weight_rule1_rapid_descent: float = 15.0
    weight_rule2_torso_horizontal: float = 15.0
    weight_rule3_aspect_ratio_lying: float = 10.0
    weight_rule4_hip_shoulder_alignment: float = 10.0
    weight_rule5_head_floor_proximity: float = 10.0
    weight_rule6_center_drop: float = 10.0
    weight_rule7_movement_then_stop: float = 10.0
    weight_rule8_prolonged_immobility: float = 10.0
    weight_rule10_medical_emergency_gate: float = 10.0   # Added when confirmation period fully satisfied

    # --- Temporal Sliding Window Configuration ---
    sliding_window_seconds: float = 15.0                 # Total duration of frame buffer stored per track
    max_fps: float = 30.0                                # Estimated max FPS to bound deque memory O(1)


@dataclass
class FrameTelemetry:
    """Instantaneous snapshot of a person's spatial and biomechanical state at timestamp t."""
    timestamp: float
    bbox: List[float]                                    # [x1, y1, x2, y2]
    keypoints: np.ndarray                                # Shape (17, 3) -> [x, y, conf]
    centroid: Tuple[float, float]                        # (cx, cy)
    bbox_height: float                                   # h = y2 - y1
    bbox_width: float                                    # w = x2 - x1
    aspect_ratio: float                                  # w / h
    torso_angle_deg: float                               # Angle of spine vector from vertical Y-axis
    spine_height_norm: float                             # |y_shoulder - y_hip| / h
    head_floor_dist_norm: float                          # (y2 - y_nose) / h
    velocity_norm: Tuple[float, float]                   # (vx/h, vy/h) instantaneous velocity per second
    speed_px_sec: float                                  # Absolute pixel speed per second


@dataclass
class RuleEvaluationReport:
    """Detailed diagnosis containing status of all 10 rules and resulting confidence tier."""
    tracker_id: int
    timestamp: float
    confidence_score: float                              # Final calculated score (0.0 - 100.0)
    confidence_level: str                                # NORMAL | POSSIBLE_FALL | HIGH_PROBABILITY_FALL | MEDICAL_EMERGENCY
    rules_triggered: Dict[str, bool]                     # Map of rule name to activation status
    rule_scores: Dict[str, float]                        # Points contributed by each active rule
    is_fallen: bool                                      # Persistent state tracking whether fall occurred
    fall_onset_time: Optional[float]                     # Timestamp when fall was first confirmed
    immobile_duration_sec: float                         # Seconds elapsed motionless after fall onset
    recovery_detected: bool                              # True if person recovered to standing posture


class SlidingWindowHistory:
    """Per-person temporal sliding window holding deterministic history across multiple seconds."""

    def __init__(self, tracker_id: int, config: MedicalEmergencyConfig) -> None:
        self.tracker_id = tracker_id
        self.config = config
        self.max_len = int(config.sliding_window_seconds * config.max_fps)
        self.history: deque[FrameTelemetry] = deque(maxlen=self.max_len)

        # Persistent fall state
        self.is_fallen: bool = False
        self.fall_onset_time: Optional[float] = None
        self.last_high_velocity_time: Optional[float] = None

    def add_frame(self, bbox: List[float], keypoints: np.ndarray, timestamp: float) -> FrameTelemetry:
        """Process raw bounding box and keypoints, compute instantaneous biomechanics, and append to window."""
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        ar = w / h

        # Extract confident keypoints
        def get_kp(idx: int) -> Optional[Tuple[float, float]]:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3 and keypoints[idx][2] >= 0.25:
                return (float(keypoints[idx][0]), float(keypoints[idx][1]))
            return None

        nose = get_kp(KP_NOSE)
        l_sh = get_kp(KP_LEFT_SHOULDER)
        r_sh = get_kp(KP_RIGHT_SHOULDER)
        l_hip = get_kp(KP_LEFT_HIP)
        r_hip = get_kp(KP_RIGHT_HIP)

        # Centroid calculation
        valid_pts = [p for p in [nose, l_sh, r_sh, l_hip, r_hip] if p is not None]
        if valid_pts:
            cx = sum(p[0] for p in valid_pts) / len(valid_pts)
            cy = sum(p[1] for p in valid_pts) / len(valid_pts)
        else:
            cx = float(bbox[0] + bbox[2]) / 2.0
            cy = float(bbox[1] + bbox[3]) / 2.0
        centroid = (cx, cy)

        # Torso inclination relative to vertical Y-axis
        torso_angle_deg = 0.0
        spine_height_norm = 1.0
        if (l_sh or r_sh) and (l_hip or r_hip):
            sh_x = (l_sh[0] if l_sh else r_sh[0] + (r_sh[0] if r_sh else l_sh[0])) / (2.0 if l_sh and r_sh else 1.0)
            sh_y = (l_sh[1] if l_sh else r_sh[1] + (r_sh[1] if r_sh else l_sh[1])) / (2.0 if l_sh and r_sh else 1.0)
            hp_x = (l_hip[0] if l_hip else r_hip[0] + (r_hip[0] if r_hip else l_hip[0])) / (2.0 if l_hip and r_hip else 1.0)
            hp_y = (l_hip[1] if l_hip else r_hip[1] + (r_hip[1] if r_hip else l_hip[1])) / (2.0 if l_hip and r_hip else 1.0)

            dx = sh_x - hp_x
            dy = sh_y - hp_y
            # Angle relative to vertical vector (0, -1) pointing up from hip to shoulder
            # arccos(|dy| / hypot(dx, dy)) gives angle from vertical
            norm = math.hypot(dx, dy)
            if norm > 1e-4:
                torso_angle_deg = math.degrees(math.acos(min(1.0, abs(dy) / norm)))
            spine_height_norm = abs(sh_y - hp_y) / h

        # Head distance to floor (bbox bottom edge y2)
        head_floor_dist_norm = 1.0
        if nose:
            head_floor_dist_norm = max(0.0, float(bbox[3]) - nose[1]) / h

        # Instantaneous velocity (normalized by bbox height per second)
        vx_norm, vy_norm, speed_px = 0.0, 0.0, 0.0
        if self.history:
            prev = self.history[-1]
            dt = max(1e-4, timestamp - prev.timestamp)
            vx_px = (cx - prev.centroid[0]) / dt
            vy_px = (cy - prev.centroid[1]) / dt
            speed_px = math.hypot(vx_px, vy_px)
            vx_norm = vx_px / h
            vy_norm = vy_px / h

        telemetry = FrameTelemetry(
            timestamp=timestamp,
            bbox=bbox,
            keypoints=keypoints,
            centroid=centroid,
            bbox_height=h,
            bbox_width=w,
            aspect_ratio=ar,
            torso_angle_deg=torso_angle_deg,
            spine_height_norm=spine_height_norm,
            head_floor_dist_norm=head_floor_dist_norm,
            velocity_norm=(vx_norm, vy_norm),
            speed_px_sec=speed_px
        )

        self.history.append(telemetry)
        return telemetry

    def get_window(self, lookback_seconds: float) -> List[FrameTelemetry]:
        """Retrieve all historical frames within `lookback_seconds` before the latest timestamp."""
        if not self.history:
            return []
        latest_t = self.history[-1].timestamp
        cutoff_t = latest_t - lookback_seconds
        return [f for f in self.history if f.timestamp >= cutoff_t]


class MedicalEmergencyAnalyzer:
    """Production-Ready Rule-Based Medical Emergency Fall Detection Engine.

    Evaluates 10 biomechanical rules over a temporal sliding window, assigns
    calibrated confidence scores, and transitions through emergency tiers without
    any deep learning classifiers.
    """

    def __init__(self, config: Optional[MedicalEmergencyConfig] = None) -> None:
        self.config = config if config is not None else MedicalEmergencyConfig()
        self.track_histories: Dict[int, SlidingWindowHistory] = {}
        logger.info("[*] MedicalEmergencyAnalyzer initialized successfully with 10 rule engines.")

    def update_config(self, new_config: MedicalEmergencyConfig) -> None:
        """Hot-swap threshold rules and weights at runtime."""
        self.config = new_config
        logger.info("[*] MedicalEmergencyAnalyzer configuration updated.")

    def analyze(
        self,
        tracker_id: int,
        bbox: List[float],
        keypoints: np.ndarray,
        timestamp: float
    ) -> RuleEvaluationReport:
        """Perform full rule-based diagnosis and return confidence assessment.

        Args:
            tracker_id: Unique ByteTrack integer ID assigned to the person.
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            keypoints: COCO-17 keypoint tensor of shape (17, 3).
            timestamp: Epoch timestamp in seconds.

        Returns:
            RuleEvaluationReport summarizing triggered rules and confidence level.
        """
        # Ensure track history sliding window exists
        if tracker_id not in self.track_histories:
            self.track_histories[tracker_id] = SlidingWindowHistory(tracker_id, self.config)
        
        window_mgr = self.track_histories[tracker_id]
        curr = window_mgr.add_frame(bbox, keypoints, timestamp)

        rules_triggered: Dict[str, bool] = {
            "rule_1_rapid_descent": False,
            "rule_2_torso_horizontal": False,
            "rule_3_aspect_ratio_lying": False,
            "rule_4_hip_shoulder_alignment": False,
            "rule_5_head_close_to_floor": False,
            "rule_6_body_center_drop": False,
            "rule_7_movement_then_stop": False,
            "rule_8_prolonged_immobility": False,
            "rule_9_recovery_detection": False,
            "rule_10_medical_emergency_trigger": False,
        }
        rule_scores: Dict[str, float] = {}

        # Retrieve temporal lookback windows
        window_1_5s = window_mgr.get_window(1.5)
        window_2_0s = window_mgr.get_window(2.0)
        window_10_0s = window_mgr.get_window(10.0)

        # ---------------------------------------------------------------------
        # RULE 1: Rapid Vertical Body Descent
        # Check instantaneous downward velocity (vy_norm > threshold)
        # ---------------------------------------------------------------------
        if curr.velocity_norm[1] >= self.config.rule1_rapid_descent_velocity_norm:
            rules_triggered["rule_1_rapid_descent"] = True
            rule_scores["rule_1_rapid_descent"] = self.config.weight_rule1_rapid_descent
            window_mgr.last_high_velocity_time = timestamp

        # ---------------------------------------------------------------------
        # RULE 2: Torso Changes from Vertical to Horizontal
        # Torso angle >= 62 deg AND was previously upright (<= 38 deg within 1.5s)
        # ---------------------------------------------------------------------
        was_upright = any(f.torso_angle_deg <= self.config.rule2_torso_upright_angle_deg for f in window_1_5s[:-1]) if len(window_1_5s) > 1 else False
        if curr.torso_angle_deg >= self.config.rule2_torso_horizontal_angle_deg and (was_upright or window_mgr.is_fallen):
            rules_triggered["rule_2_torso_horizontal"] = True
            rule_scores["rule_2_torso_horizontal"] = self.config.weight_rule2_torso_horizontal

        # ---------------------------------------------------------------------
        # RULE 3: Bounding Box Aspect Ratio Changes from Standing to Lying
        # AR >= 1.05 AND was previously standing (AR <= 0.68 within 2.0s)
        # ---------------------------------------------------------------------
        was_standing_ar = any(f.aspect_ratio <= self.config.rule3_aspect_ratio_standing for f in window_2_0s[:-1]) if len(window_2_0s) > 1 else False
        if curr.aspect_ratio >= self.config.rule3_aspect_ratio_lying and (was_standing_ar or window_mgr.is_fallen):
            rules_triggered["rule_3_aspect_ratio_lying"] = True
            rule_scores["rule_3_aspect_ratio_lying"] = self.config.weight_rule3_aspect_ratio_lying

        # ---------------------------------------------------------------------
        # RULE 4: Hip and Shoulder Alignment Indicates Lying Posture
        # Vertical distance between shoulder midpoint and hip midpoint <= 0.28 * h
        # ---------------------------------------------------------------------
        if curr.spine_height_norm <= self.config.rule4_hip_shoulder_alignment_norm:
            rules_triggered["rule_4_hip_shoulder_alignment"] = True
            rule_scores["rule_4_hip_shoulder_alignment"] = self.config.weight_rule4_hip_shoulder_alignment

        # ---------------------------------------------------------------------
        # RULE 5: Head Moves Close to the Floor
        # Distance between nose and bbox bottom edge y2 <= 0.22 * h
        # ---------------------------------------------------------------------
        if curr.head_floor_dist_norm <= self.config.rule5_head_floor_proximity_norm:
            rules_triggered["rule_5_head_close_to_floor"] = True
            rule_scores["rule_5_head_close_to_floor"] = self.config.weight_rule5_head_floor_proximity

        # ---------------------------------------------------------------------
        # RULE 6: Body Center Suddenly Drops
        # Net vertical drop of centroid from the average 1.5s ago >= 0.32 * h
        # ---------------------------------------------------------------------
        if len(window_1_5s) >= 3:
            oldest_avg_cy = sum(f.centroid[1] for f in window_1_5s[:3]) / min(3, len(window_1_5s[:3]))
            net_drop_norm = (curr.centroid[1] - oldest_avg_cy) / curr.bbox_height
            if net_drop_norm >= self.config.rule6_center_drop_norm:
                rules_triggered["rule_6_body_center_drop"] = True
                rule_scores["rule_6_body_center_drop"] = self.config.weight_rule6_center_drop

        # ---------------------------------------------------------------------
        # RULE 7: Large Movement Followed by an Abrupt Stop
        # High velocity observed within past 2s followed by near-zero velocity (< 0.05) now
        # ---------------------------------------------------------------------
        had_high_vel = any(math.hypot(f.velocity_norm[0], f.velocity_norm[1]) >= self.config.rule7_high_velocity_threshold_norm for f in window_2_0s)
        curr_vel_mag = math.hypot(curr.velocity_norm[0], curr.velocity_norm[1])
        if had_high_vel and curr_vel_mag <= self.config.rule7_abrupt_stop_velocity_norm:
            rules_triggered["rule_7_movement_then_stop"] = True
            rule_scores["rule_7_movement_then_stop"] = self.config.weight_rule7_movement_then_stop

        # ---------------------------------------------------------------------
        # RULE 8: Very Low Body Movement for 5–10 Seconds After the Fall
        # Evaluates average pixel speed during the [5.0s, 10.0s] post-fall window
        # Excludes initial impact frames (first 0.5s after onset) to measure true immobility
        # ---------------------------------------------------------------------
        immobile_duration = 0.0
        if window_mgr.is_fallen and window_mgr.fall_onset_time is not None:
            immobile_duration = timestamp - window_mgr.fall_onset_time
            if 5.0 <= immobile_duration <= 10.0:
                # Check average speed across frames strictly after the collapse impact (+0.5s)
                window_post_fall = [f for f in window_10_0s if f.timestamp > window_mgr.fall_onset_time + 0.5]
                if not window_post_fall:
                    window_post_fall = [curr]
                avg_speed = sum(f.speed_px_sec for f in window_post_fall) / len(window_post_fall)
                if avg_speed <= self.config.rule8_immobility_speed_px_sec:
                    rules_triggered["rule_8_prolonged_immobility"] = True
                    rule_scores["rule_8_prolonged_immobility"] = self.config.weight_rule8_prolonged_immobility

        # ---------------------------------------------------------------------
        # FALL CONFIRMATION STATE MACHINE
        # Trigger High Probability Fall when core rules confirm collapse
        # ---------------------------------------------------------------------
        raw_score_before_recovery = sum(rule_scores.values())
        if not window_mgr.is_fallen and raw_score_before_recovery >= 40.0:
            # Check if posture confirms lying on ground (Rules 2, 3, 4, or 5)
            if any([rules_triggered["rule_2_torso_horizontal"],
                    rules_triggered["rule_3_aspect_ratio_lying"],
                    rules_triggered["rule_4_hip_shoulder_alignment"],
                    rules_triggered["rule_5_head_close_to_floor"]]):
                window_mgr.is_fallen = True
                window_mgr.fall_onset_time = timestamp
                logger.warning(f"[!] Fall Onset Confirmed for Track ID {tracker_id} at t={timestamp:.2f}s (Score: {raw_score_before_recovery:.1f})")

        # ---------------------------------------------------------------------
        # RULE 9: Detect Whether the Person Recovers (Stands Up)
        # Torso returns to upright (<= 40 deg), AR <= 0.70, upward velocity (< 0)
        # If recovery occurs, immediately cancel the emergency and reset state.
        # ---------------------------------------------------------------------
        recovery_detected = False
        if window_mgr.is_fallen:
            if (curr.torso_angle_deg <= self.config.rule9_recovery_torso_angle_deg and
                curr.aspect_ratio <= self.config.rule9_recovery_aspect_ratio and
                curr.velocity_norm[1] < -0.05):  # Moving upwards
                
                recovery_detected = True
                rules_triggered["rule_9_recovery_detection"] = True
                logger.info(f"[*] Recovery Detected for Track ID {tracker_id}! Canceling emergency state.")
                
                # Reset persistent fall state
                window_mgr.is_fallen = False
                window_mgr.fall_onset_time = None
                immobile_duration = 0.0

        # ---------------------------------------------------------------------
        # RULE 10: Trigger Medical Emergency Only if Lying & Motionless After Confirmation Period
        # Requires persistent lying + motionless state past confirmation duration (>= 5.0s)
        # ---------------------------------------------------------------------
        if window_mgr.is_fallen and not recovery_detected and window_mgr.fall_onset_time is not None:
            immobile_duration = timestamp - window_mgr.fall_onset_time
            if immobile_duration >= self.config.rule10_confirmation_period_sec:
                # Check recent movement over the confirmation period post-impact (+0.5s)
                recent_frames = [f for f in window_10_0s if f.timestamp > window_mgr.fall_onset_time + 0.5]
                if not recent_frames:
                    recent_frames = [curr]
                avg_speed = sum(f.speed_px_sec for f in recent_frames) / len(recent_frames)
                if avg_speed <= self.config.rule8_immobility_speed_px_sec:
                    rules_triggered["rule_10_medical_emergency_trigger"] = True
                    rule_scores["rule_10_medical_emergency_trigger"] = self.config.weight_rule10_medical_emergency_gate

        # ---------------------------------------------------------------------
        # CONFIDENCE TIER CALCULATION
        # Sum rule weights and determine final tier (0-100)
        # ---------------------------------------------------------------------
        if recovery_detected:
            final_score = 0.0
        else:
            final_score = min(100.0, sum(rule_scores.values()))
            # If persistent fall confirmed and lying flat, elevate baseline to at least 70
            if window_mgr.is_fallen and any([rules_triggered["rule_2_torso_horizontal"], rules_triggered["rule_3_aspect_ratio_lying"]]):
                final_score = max(final_score, 72.0)
            # If Rule 10 triggered, elevate strictly to Medical Emergency (90-100)
            if rules_triggered["rule_10_medical_emergency_trigger"]:
                final_score = max(final_score, 94.0)

        # Map score to requested tier boundaries
        if final_score >= 90.0:
            confidence_level = "MEDICAL_EMERGENCY"
        elif final_score >= 70.0:
            confidence_level = "HIGH_PROBABILITY_FALL"
        elif final_score >= 40.0:
            confidence_level = "POSSIBLE_FALL"
        else:
            confidence_level = "NORMAL"

        return RuleEvaluationReport(
            tracker_id=tracker_id,
            timestamp=timestamp,
            confidence_score=round(final_score, 1),
            confidence_level=confidence_level,
            rules_triggered=rules_triggered,
            rule_scores=rule_scores,
            is_fallen=window_mgr.is_fallen,
            fall_onset_time=window_mgr.fall_onset_time,
            immobile_duration_sec=round(immobile_duration, 1),
            recovery_detected=recovery_detected
        )

    def prune_stale_tracks(self, active_track_ids: Set[int], timeout_seconds: float = 30.0) -> None:
        """Remove historical sliding windows for individuals who have left the scene."""
        stale = [
            tid for tid, mgr in self.track_histories.items()
            if tid not in active_track_ids and (mgr.history and (mgr.history[-1].timestamp + timeout_seconds < mgr.history[-1].timestamp))
        ]
        for tid in stale:
            del self.track_histories[tid]
