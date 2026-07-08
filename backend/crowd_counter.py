import time
from dataclasses import dataclass
from typing import Dict, Set, Optional, Tuple
import cv2
import numpy as np
import supervision as sv


@dataclass(frozen=True)
class CrowdMetrics:
    """Immutable telemetry data emitted by CrowdCounter per frame."""
    current_count: int          # Active unique people in the current frame
    cumulative_count: int       # Total unique Track IDs seen since startup/reset
    peak_count: int             # Maximum simultaneous occupancy observed
    active_track_ids: Set[int]  # Set of currently active ByteTrack IDs


class CrowdCounter:
    """Real-time crowd occupancy and visitor analytics module.

    Consumes ByteTrack tracked detections to monitor active crowd density,
    prevents duplicate counting across frames using persistent set hashing,
    and renders a professional dashboard overlay onto video frames.
    """

    def __init__(self, count_cooldown_seconds: float = 1.0) -> None:
        self._active_ids: Set[int] = set()
        self._cumulative_ids: Set[int] = set()
        self._peak_occupancy: int = 0
        self._last_update_time: float = 0.0
        self._count_cooldown = count_cooldown_seconds
        
        # Track entry timestamps to prevent transient flickering false-positives
        self._id_first_seen: Dict[int, float] = {}

    def reset(self) -> None:
        """Reset all occupancy counters and historical telemetry."""
        self._active_ids.clear()
        self._cumulative_ids.clear()
        self._id_first_seen.clear()
        self._peak_occupancy = 0

    def update(
        self, 
        detections: sv.Detections, 
        current_time: Optional[float] = None
    ) -> CrowdMetrics:
        """Update crowd metrics from current frame ByteTrack detections.

        Args:
            detections: supervision.Detections object containing `tracker_id`.
            current_time: Epoch timestamp (defaults to time.time()).

        Returns:
            CrowdMetrics snapshot for WebSocket streaming and logging.
        """
        now = current_time if current_time is not None else time.time()
        self._last_update_time = now

        # 1. Extract valid active ByteTrack integer IDs
        current_active: Set[int] = set()
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for tid in detections.tracker_id:
                tid_int = int(tid)
                current_active.add(tid_int)
                
                # Record initial sighting timestamp for new IDs
                if tid_int not in self._id_first_seen:
                    self._id_first_seen[tid_int] = now

        # 2. Prevent Duplicate Counting: Only add to cumulative after ID stabilizes (e.g. >= 0.2s)
        for tid_int in current_active:
            if now - self._id_first_seen[tid_int] >= 0.2:
                self._cumulative_ids.add(tid_int)

        # 3. Update internal state
        self._active_ids = current_active
        current_count = len(self._active_ids)
        if current_count > self._peak_occupancy:
            self._peak_occupancy = current_count

        # 4. Clean up stale sighting timestamps for IDs unseen for > 60 seconds
        stale_tids = [
            tid for tid, seen_time in self._id_first_seen.items() 
            if tid not in self._active_ids and (now - seen_time) > 60.0
        ]
        for tid in stale_tids:
            del self._id_first_seen[tid]

        return CrowdMetrics(
            current_count=current_count,
            cumulative_count=len(self._cumulative_ids),
            peak_count=self._peak_occupancy,
            active_track_ids=self._active_ids.copy(),
        )

    def draw_hud(
        self, 
        frame: np.ndarray, 
        metrics: Optional[CrowdMetrics] = None,
        position: Tuple[int, int] = (20, 40)
    ) -> np.ndarray:
        """Render a high-visibility Crowd Analytics HUD onto the frame."""
        if metrics is None:
            metrics = CrowdMetrics(
                current_count=len(self._active_ids),
                cumulative_count=len(self._cumulative_ids),
                peak_count=self._peak_occupancy,
                active_track_ids=self._active_ids.copy()
            )

        x, y = position
        hud_width, hud_height = 280, 90

        # Draw semi-transparent dark background overlay for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 25), (x + hud_width, y + hud_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw border highlight
        cv2.rectangle(frame, (x - 10, y - 25), (x + hud_width, y + hud_height), (56, 189, 248), 2)

        # Render Metrics Typography
        cv2.putText(
            frame, f"ACTIVE CROWD: {metrics.current_count}", 
            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Peak Occupancy: {metrics.peak_count}", 
            (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )
        cv2.putText(
            frame, f"Total Unique Visitors: {metrics.cumulative_count}", 
            (x, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
        )

        return frame
