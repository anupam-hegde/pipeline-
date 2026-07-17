"""YOLOX and RTMPose inference adapters for real-time CCTV surveillance.

Provides clean, typed boundaries around an ONNX-runtime YOLOX (object
detection) and MMPose RTMPose (pose estimation) so that all application
logic — ByteTrack tracking, kinematics, WebSocket streaming — stays fully
independent of the underlying model framework. Object detection now
runs directly on ONNX Runtime; no MMDetection/MMEngine dependency is
required at inference time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import supervision as sv


COCO17_KEYPOINTS = 17
PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PoseDetectionBatch:
    """Per-frame person detections with COCO-17 keypoints and optional ByteTrack IDs."""

    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    keypoints: np.ndarray
    tracker_id: Optional[np.ndarray] = None

    @classmethod
    def empty(cls) -> "PoseDetectionBatch":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
            keypoints=np.empty((0, COCO17_KEYPOINTS, 3), dtype=np.float32),
            tracker_id=np.empty((0,), dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def to_supervision(self) -> sv.Detections:
        return sv.Detections(
            xyxy=self.xyxy,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
        )


@dataclass(frozen=True)
class ObjectDetectionBatch:
    """Generic object detector output in xyxy format with optional ByteTrack IDs."""

    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: Optional[np.ndarray] = None

    @classmethod
    def empty(cls) -> "ObjectDetectionBatch":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
            tracker_id=np.empty((0,), dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def to_supervision(self) -> sv.Detections:
        return sv.Detections(
            xyxy=self.xyxy,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
        )


class RTMPosePipeline:
    """Top-down RTMPose pipeline backed by the MMPose inferencer.

    The default MMPose ``human`` alias is RTMPose with an RTMDet human
    detector. Config and checkpoint paths can be overridden with
    environment variables or constructor arguments for offline deployments.
    """

    def __init__(
        self,
        pose_model: str = "human",
        pose_weights: Optional[str] = None,
        det_model: str = "human",
        det_weights: Optional[str] = None,
        device: Optional[str] = None,
        det_cat_ids: Optional[Iterable[int]] = None,
    ) -> None:
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise RuntimeError(
                "RTMPose inference requires mmpose, mmdet, mmcv, and mmengine. "
                "Install the OpenMMLab dependencies from requirements.txt."
            ) from exc

        effective_det_model = None if det_model in ("human", "default", None) else det_model
        self._inferencer = MMPoseInferencer(
            pose2d=pose_model,
            pose2d_weights=pose_weights,
            det_model=effective_det_model,
            det_weights=det_weights,
            det_cat_ids=list(det_cat_ids or [PERSON_CLASS_ID]),
            device=device,
        )

    @classmethod
    def from_env(cls) -> "RTMPosePipeline":
        return cls(
            pose_model=os.getenv("RTMPOSE_MODEL", "human"),
            pose_weights=os.getenv("RTMPOSE_WEIGHTS") or None,
            det_model=os.getenv("RTMPOSE_DET_MODEL", "human"),
            det_weights=os.getenv("RTMPOSE_DET_WEIGHTS") or None,
            device=os.getenv("CV_DEVICE") or None,
        )

    def predict(
        self, 
        frame: np.ndarray, 
        confidence: float = 0.5,
        tracked_boxes: Optional[sv.Detections] = None,
    ) -> PoseDetectionBatch:
        inferencer_kwargs = {
            "show": False,
            "return_vis": False,
            "kpt_thr": confidence,
        }
        if tracked_boxes is not None and len(tracked_boxes) > 0:
            inferencer_kwargs["bboxes"] = tracked_boxes.xyxy.tolist()

        result_generator = self._inferencer(frame, **inferencer_kwargs)
        result = next(result_generator)
        predictions = result.get("predictions", [])
        instances = predictions[0] if predictions else []
        batch = _parse_pose_instances(instances, confidence)

        if tracked_boxes is not None and len(tracked_boxes) == len(batch):
            return PoseDetectionBatch(
                xyxy=batch.xyxy,
                confidence=batch.confidence,
                class_id=batch.class_id,
                keypoints=batch.keypoints,
                tracker_id=tracked_boxes.tracker_id.astype(np.int32) if tracked_boxes.tracker_id is not None else None,
            )
        return batch


class YOLOXObjectDetector:
    """YOLOX detector wrapper for custom fire/weapon detection.

    Runs YOLOX-s ONNX export via ONNX Runtime (TensorRT/CUDA). All
    coordinate outputs are in xyxy pixel format. Class IDs follow the
    project schema defined during training: 0=fire, 1=weapon.

    If the ONNX model is not found, ``from_env`` returns ``None`` so
    callers silently fall back to pose-only mode.
    """

    def __init__(self, onnx_path: str, device: Optional[str] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "YOLOX detection requires onnxruntime-gpu. Install it from requirements.txt."
            ) from exc

        self._onnx_path = Path(onnx_path)
        self._input_size = (640, 640)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device and "cuda" in device.lower()
            else ["CPUExecutionProvider"]
        )
        session_opts = ort.SessionOptions()
        session_opts.log_severity_level = 3
        session_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        try:
            self._session = ort.InferenceSession(
                str(self._onnx_path),
                sess_options=session_opts,
                providers=providers,
            )
        except Exception:
            self._session = ort.InferenceSession(
                str(self._onnx_path),
                sess_options=session_opts,
                providers=["CPUExecutionProvider"],
            )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @classmethod
    def from_env(cls) -> Optional["YOLOXObjectDetector"]:
        onnx_path = os.getenv(
            "YOLOX_ONNX_MODEL",
            "models/yolox_surveillance.onnx",
        )
        if not Path(onnx_path).exists():
            return None
        return cls(onnx_path=onnx_path, device=os.getenv("CV_DEVICE") or None)

    def predict(
        self, frame: np.ndarray, confidence: float = 0.45
    ) -> ObjectDetectionBatch:
        letterboxed, scale, pad_left, pad_top = _letterbox(
            frame, self._input_size
        )
        blob = letterboxed.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        ort_inputs = {self._input_name: blob}
        raw = self._session.run([self._output_name], ort_inputs)[0]

        return _decode_yolox_output(
            raw, scale, pad_left, pad_top, frame.shape[:2], confidence
        )


def create_pose_pipeline() -> RTMPosePipeline:
    return RTMPosePipeline.from_env()


def create_object_detector() -> Optional[YOLOXObjectDetector]:
    return YOLOXObjectDetector.from_env()


# Compatibility layer: alias RTMDetObjectDetector to YOLOXObjectDetector so legacy imports continue to work
RTMDetObjectDetector = YOLOXObjectDetector


# ---------------------------------------------------------------------------
# YOLOX ONNX helper: letterbox + output decoding
# ---------------------------------------------------------------------------

YOLOX_NUM_CLASSES = 2
YOLOX_STRIDES = (8, 16, 32)


def _letterbox(
    img: np.ndarray, target_size: tuple[int, int]
) -> tuple[np.ndarray, float, int, int]:
    ih, iw = img.shape[:2]
    th, tw = target_size
    scale = min(th / ih, tw / iw)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = tw - new_w
    pad_h = th - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, pad_left, pad_top


def _decode_yolox_output(
    raw: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    frame_shape: tuple[int, int],
    conf_thr: float,
) -> ObjectDetectionBatch:
    nms_thr = 0.45
    fh, fw = frame_shape
    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []
    all_classes: list[int] = []

    offset = 0
    for stride in YOLOX_STRIDES:
        grid_h = 640 // stride
        grid_w = 640 // stride
        per_cell = YOLOX_NUM_CLASSES + 5
        cells = grid_h * grid_w
        block = raw[:, offset : offset + cells * per_cell].reshape(-1, per_cell)
        offset += cells * per_cell

        obj_conf = 1.0 / (1.0 + np.exp(-block[:, 4]))
        cls_logits = block[:, 5 : 5 + YOLOX_NUM_CLASSES]
        cls_scores = 1.0 / (1.0 + np.exp(-cls_logits)) * obj_conf[:, np.newaxis]
        best_cls = np.argmax(cls_scores, axis=1)
        best_score = np.max(cls_scores, axis=1)
        keep_map = best_score > conf_thr
        if not np.any(keep_map):
            continue

        keep_idx = np.where(keep_map)[0]
        scores = best_score[keep_idx]
        classes = best_cls[keep_idx].astype(np.int32)

        row = keep_idx // grid_w
        col = keep_idx % grid_w
        box_raw = block[keep_idx, :4]

        cx_pred = (col + 1.0 / (1.0 + np.exp(-box_raw[:, 0]))) * stride
        cy_pred = (row + 1.0 / (1.0 + np.exp(-box_raw[:, 1]))) * stride
        w_pred = np.exp(box_raw[:, 2]) * stride
        h_pred = np.exp(box_raw[:, 3]) * stride

        x1 = (cx_pred - w_pred / 2.0 - pad_left) / scale
        y1 = (cy_pred - h_pred / 2.0 - pad_top) / scale
        x2 = (cx_pred + w_pred / 2.0 - pad_left) / scale
        y2 = (cy_pred + h_pred / 2.0 - pad_top) / scale

        x1 = np.clip(x1, 0.0, float(fw))
        y1 = np.clip(y1, 0.0, float(fh))
        x2 = np.clip(x2, 0.0, float(fw))
        y2 = np.clip(y2, 0.0, float(fh))

        for i in range(len(scores)):
            bw = x2[i] - x1[i]
            bh = y2[i] - y1[i]
            if bw < 1.0 or bh < 1.0:
                continue
            all_boxes.append(np.array([x1[i], y1[i], x2[i], y2[i]], dtype=np.float32))
            all_scores.append(float(scores[i]))
            all_classes.append(int(classes[i]))

    if not all_boxes:
        return ObjectDetectionBatch.empty()

    boxes_np = np.stack(all_boxes)
    scores_np = np.array(all_scores, dtype=np.float32)
    classes_np = np.array(all_classes, dtype=np.int32)

    keep_idx = _nms(boxes_np, scores_np, nms_thr)
    if len(keep_idx) == 0:
        return ObjectDetectionBatch.empty()

    return ObjectDetectionBatch(
        xyxy=boxes_np[keep_idx],
        confidence=scores_np[keep_idx],
        class_id=classes_np[keep_idx],
    )


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return np.array(keep, dtype=np.int32)


def match_keypoints_to_bbox(
    pose_batch: PoseDetectionBatch,
    tracked_bbox: np.ndarray,
    min_iou: float = 0.01,
) -> Optional[np.ndarray]:
    """Find the pose keypoints corresponding to a tracked bounding box."""

    if len(pose_batch) == 0:
        return None

    ious = np.array([bbox_iou(tracked_bbox, box) for box in pose_batch.xyxy])
    best_idx = int(np.argmax(ious))
    if float(ious[best_idx]) < min_iou:
        return None

    return pose_batch.keypoints[best_idx]


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def draw_object_detections(frame: np.ndarray, detections: ObjectDetectionBatch) -> None:
    """Draw fire/weapon outputs from the custom object detector."""

    for bbox, score, class_id in zip(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
    ):
        if int(class_id) not in {0, 1}:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        if int(class_id) == 0:
            label = f"FIRE WARNING {float(score):.2f}"
            color = (0, 165, 255)
        else:
            label = f"WEAPON DETECTED {float(score):.2f}"
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def _parse_pose_instances(
    instances: list[dict],
    min_score: float,
) -> PoseDetectionBatch:
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    keypoints: list[np.ndarray] = []

    for instance in instances:
        kps = _to_coco17_keypoints(instance)
        if kps is None:
            continue

        bbox = _extract_bbox(instance, kps)
        score = _extract_score(instance, kps)
        if score < min_score:
            continue

        boxes.append(bbox)
        scores.append(score)
        keypoints.append(kps)

    if not boxes:
        return PoseDetectionBatch.empty()

    return PoseDetectionBatch(
        xyxy=np.vstack(boxes).astype(np.float32),
        confidence=np.array(scores, dtype=np.float32),
        class_id=np.full((len(boxes),), PERSON_CLASS_ID, dtype=np.int32),
        keypoints=np.stack(keypoints).astype(np.float32),
    )


def _to_coco17_keypoints(instance: dict) -> Optional[np.ndarray]:
    raw_keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    if raw_keypoints.ndim != 2 or raw_keypoints.shape[0] < COCO17_KEYPOINTS:
        return None

    raw_scores = instance.get("keypoint_scores")
    if raw_scores is None:
        scores = np.ones((raw_keypoints.shape[0], 1), dtype=np.float32)
    else:
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1, 1)

    if raw_keypoints.shape[1] >= 3:
        combined = raw_keypoints[:, :3]
    else:
        combined = np.concatenate([raw_keypoints[:, :2], scores], axis=1)

    return combined[:COCO17_KEYPOINTS]


def _extract_bbox(instance: dict, keypoints: np.ndarray) -> np.ndarray:
    raw_bbox = instance.get("bbox")
    if raw_bbox is not None:
        bbox = np.asarray(raw_bbox, dtype=np.float32).reshape(-1)
        if bbox.size >= 4:
            return bbox[:4]

    valid = keypoints[keypoints[:, 2] > 0.0]
    if valid.size == 0:
        return np.zeros((4,), dtype=np.float32)

    x1, y1 = np.min(valid[:, :2], axis=0)
    x2, y2 = np.max(valid[:, :2], axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _extract_score(instance: dict, keypoints: np.ndarray) -> float:
    for key in ("bbox_score", "score"):
        raw_score = instance.get(key)
        if raw_score is None:
            continue
        score = np.asarray(raw_score, dtype=np.float32).reshape(-1)
        if score.size:
            return float(score[0])

    valid_scores = keypoints[:, 2]
    valid_scores = valid_scores[valid_scores > 0.0]
    if valid_scores.size == 0:
        return 0.0

    return float(np.mean(valid_scores))
