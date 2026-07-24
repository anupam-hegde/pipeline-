from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Set
import cv2
import numpy as np
import supervision as sv


class DensityLevel(str, Enum):
    """Crowd density classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class DensityThresholds:
    """Configurable threshold rules for crowd density classification.
    
    A scene is classified into a higher density tier if EITHER the occupied
    area ratio OR the active person headcount exceeds the threshold.
    """
    medium_area_ratio: float = 0.15     # 15% of frame pixels occupied by people
    high_area_ratio: float = 0.35       # 35% of frame pixels occupied
    medium_person_count: int = 5        # At least 5 active tracked individuals
    high_person_count: int = 12         # At least 12 active tracked individuals


@dataclass(frozen=True)
class DensityMetrics:
    """Immutable density telemetry emitted per frame."""
    level: DensityLevel
    occupied_area_ratio: float          # Normalized ratio [0.0, 1.0]
    occupied_pixels: int                # True union of occupied pixels
    total_frame_pixels: int             # Total resolution area (H * W)
    active_person_count: int            # Number of valid ByteTrack tracks
    thresholds_applied: DensityThresholds


class CrowdDensityEstimator:
    """Production-ready crowd density estimation module.

    Computes true spatial occupancy by calculating the geometric union of all
    tracked bounding boxes (preventing double-counting from occlusions), and
    classifies scene density based on configurable multi-factor thresholds.
    """

    def __init__(self, thresholds: Optional[DensityThresholds] = None) -> None:
        self.thresholds = thresholds if thresholds is not None else DensityThresholds()
        self._last_metrics: Optional[DensityMetrics] = None

    def update_thresholds(self, new_thresholds: DensityThresholds) -> None:
        """Hot-swap density configuration thresholds at runtime."""
        self.thresholds = new_thresholds

    def estimate(
        self, 
        frame_shape: Tuple[int, int, ...], 
        detections: sv.Detections
    ) -> DensityMetrics:
        """Calculate crowd density and classify occupancy level.

        Args:
            frame_shape: Tuple of (Height, Width, Channels) from cv2 image.
            detections: supervision.Detections containing active ByteTrack bboxes.

        Returns:
            DensityMetrics snapshot with classification and spatial telemetry.
        """
        frame_height, frame_width = frame_shape[:2]
        total_pixels = max(1, frame_height * frame_width)

        # 1. Filter for active ByteTrack tracks only
        active_boxes: list[np.ndarray] = []
        active_ids: Set[int] = set()
        
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for idx, bbox in enumerate(detections.xyxy):
                # Ensure box has a valid ByteTrack ID assigned
                if detections.tracker_id is not None:
                    tid = int(detections.tracker_id[idx])
                    if tid < 0 or tid in active_ids:
                        continue
                    active_ids.add(tid)
                active_boxes.append(bbox)

        person_count = len(active_boxes)
        if person_count == 0:
            self._last_metrics = DensityMetrics(
                level=DensityLevel.LOW,
                occupied_area_ratio=0.0,
                occupied_pixels=0,
                total_frame_pixels=total_pixels,
                active_person_count=0,
                thresholds_applied=self.thresholds,
            )
            return self._last_metrics

        # 2. Occlusion-Aware Area Union Calculation
        # Render bounding boxes onto a downsampled 2D boolean occupancy grid to get exact union area
        scale = 0.25
        grid_h, grid_w = int(frame_height * scale), int(frame_width * scale)
        occupancy_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for bbox in active_boxes:
            x1, y1, x2, y2 = bbox * scale
            x1_idx = np.clip(int(x1), 0, grid_w)
            y1_idx = np.clip(int(y1), 0, grid_h)
            x2_idx = np.clip(int(np.ceil(x2)), 0, grid_w)
            y2_idx = np.clip(int(np.ceil(y2)), 0, grid_h)
            
            if x2_idx > x1_idx and y2_idx > y1_idx:
                occupancy_mask[y1_idx:y2_idx, x1_idx:x2_idx] = 1

        occupied_grid_pixels = int(np.sum(occupancy_mask))
        occupied_area_ratio = float(occupied_grid_pixels) / float(max(1, grid_h * grid_w))
        approx_full_pixels = int(occupied_area_ratio * total_pixels)

        # 3. Multi-Factor Classification
        level = DensityLevel.LOW
        if (occupied_area_ratio >= self.thresholds.high_area_ratio or 
            person_count >= self.thresholds.high_person_count):
            level = DensityLevel.HIGH
        elif (occupied_area_ratio >= self.thresholds.medium_area_ratio or 
              person_count >= self.thresholds.medium_person_count):
            level = DensityLevel.MEDIUM

        self._last_metrics = DensityMetrics(
            level=level,
            occupied_area_ratio=occupied_area_ratio,
            occupied_pixels=approx_full_pixels,
            total_frame_pixels=total_pixels,
            active_person_count=person_count,
            thresholds_applied=self.thresholds,
        )
        return self._last_metrics

    def draw_hud(
        self, 
        frame: np.ndarray, 
        metrics: Optional[DensityMetrics] = None,
        position: Tuple[int, int] = (15, 80)
    ) -> np.ndarray:
        """Render a compact, color-coded Crowd Density badge onto the video stream."""
        if metrics is None:
            metrics = self._last_metrics
        if metrics is None:
            return frame

        x, y = position
        
        # Color palettes (BGR): Green for Low, Yellow/Orange for Medium, Red for High
        colors = {
            DensityLevel.LOW: (74, 222, 128),       # Bright Green
            DensityLevel.MEDIUM: (0, 165, 255),     # Orange/Yellow
            DensityLevel.HIGH: (0, 0, 255),         # Alert Red
        }
        active_color = colors.get(metrics.level, (255, 255, 255))

        hud_width = 186
        hud_height = 58 if metrics.level == DensityLevel.HIGH else 42

        # Render Background Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 6, y - 16), (x - 6 + hud_width, y - 16 + hud_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x - 6, y - 16), (x - 6 + hud_width, y - 16 + hud_height), active_color, 1)

        # Draw Level Badge & Telemetry (compact 2 lines)
        cv2.putText(
            frame, f"DENSITY: {metrics.level.value}", 
            (x, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1
        )
        cv2.putText(
            frame, f"Area: {metrics.occupied_area_ratio * 100:.1f}% | Tracks: {metrics.active_person_count}", 
            (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1
        )

        # Draw Compact Warning Line if High Density
        if metrics.level == DensityLevel.HIGH:
            cv2.putText(
                frame, "HIGH DENSITY DETECTED", 
                (x, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1
            )

        return frame
