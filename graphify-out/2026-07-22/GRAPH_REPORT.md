# Graph Report - pipeline  (2026-07-22)

## Corpus Check
- 22 files · ~906,491 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 275 nodes · 492 edges · 18 communities (13 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `17b2102b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Vision Models & BBox Processing
- YOLOX Surveillance Configuration
- YOLO to COCO Conversion
- Action Math & Violence Detection
- FastAPI WebSocket & Webcam Server
- Dataset Preparation & Directory Creation
- video_websocket
- Directory Video Testing & Extraction
- diagnostic_run.py
- Fall Detection & Emergency Monitoring
- RTMPose Pipeline Inference
- video_websocket
- Graphify AI Agent Rules
- Graphify Workflow Automation
- YOLOX Training Log History
- Project Python Dependencies
- .draw_hud

## God Nodes (most connected - your core abstractions)
1. `PersonHistoryManager` - 17 edges
2. `detect_advanced_violence()` - 14 edges
3. `MedicalEmergencyAnalyzer` - 14 edges
4. `ObjectDetectionBatch` - 14 edges
5. `create_pose_pipeline()` - 13 edges
6. `create_object_detector()` - 13 edges
7. `PersonTrackHistory` - 12 edges
8. `match_keypoints_to_bbox()` - 12 edges
9. `draw_object_detections()` - 12 edges
10. `Exp` - 12 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `Exp`  [EXTRACTED]
  scripts/convert_best_to_onnx.py → configs/yolox_surveillance.py
- `main()` --calls--> `Exp`  [EXTRACTED]
  scripts/train_model.py → configs/yolox_surveillance.py
- `export_and_verify()` --calls--> `Exp`  [EXTRACTED]
  scripts/export_onnx.py → configs/yolox_surveillance.py
- `check_violence_between_pairs()` --calls--> `detect_advanced_violence()`  [EXTRACTED]
  backend/main.py → backend/action_math.py
- `main()` --calls--> `detect_fall()`  [EXTRACTED]
  backend/diagnostic_run.py → backend/action_math.py

## Import Cycles
- None detected.

## Communities (18 total, 5 thin omitted)

### Community 0 - "Vision Models & BBox Processing"
Cohesion: 0.11
Nodes (21): bbox_iou(), _decode_onnx_detector_output(), _decode_yolox_output(), _extract_bbox(), _extract_score(), _letterbox(), match_keypoints_to_bbox(), _nms() (+13 more)

### Community 1 - "YOLOX Surveillance Configuration"
Cohesion: 0.15
Nodes (11): BaseExp, Exp, yolox_surveillance.py — YOLOX-s Experiment Config for Surveillance Detection  Mo, Override to point images at merged_dataset/val/images/<file_name>, Override to point images at merged_dataset/train/images/<file_name>         YOLO, main(), convert_best_to_onnx.py — Export best4.pt to best.onnx and compare PyTorch vs ON, export_and_verify() (+3 more)

### Community 2 - "YOLO to COCO Conversion"
Cohesion: 0.13
Nodes (22): convert_dataset_root(), convert_split_to_coco(), get_image_info(), main(), Any, Path, convert_yolo_to_coco.py — Fast Parallel YOLO to COCO JSON Annotation Converter, Converts standard train/val splits under dataset_root to COCO JSONs. (+14 more)

### Community 3 - "Action Math & Violence Detection"
Cohesion: 0.09
Nodes (41): detect_advanced_violence(), detect_fall(), detect_immobility(), extract_keypoint(), action_math.py — Production-Grade Kinematics Module for Violence & Fall Detectio, Determines if Person A is striking Person B using weighted     confidence scorin, Detects a fall using weighted confidence scoring across 6 independent     signal, Detects whether a tracked person is immobile (not moving) by     measuring centr (+33 more)

### Community 4 - "FastAPI WebSocket & Webcam Server"
Cohesion: 0.17
Nodes (10): DensityLevel, DensityMetrics, Detections, ndarray, Crowd density classification levels., Render a compact, color-coded Crowd Density badge onto the video stream., Immutable density telemetry emitted per frame., Calculate crowd density and classify occupancy level.          Args: (+2 more)

### Community 5 - "Dataset Preparation & Directory Creation"
Cohesion: 0.36
Nodes (8): create_directories(), generate_yaml(), main(), process_dataset(), Path, Generates the data.yaml configuration file for YOLO training., Creates the YOLO directory structure inside OUTPUT_DIR., Scans a source dataset (either direct images/labels or split train/val/test fold

### Community 6 - "video_websocket"
Cohesion: 0.13
Nodes (13): FrameTelemetry, MedicalEmergencyConfig, ndarray, Detailed diagnosis containing status of all 10 rules and resulting confidence ti, Per-person temporal sliding window holding deterministic history across multiple, Process raw bounding box and keypoints, compute instantaneous biomechanics, and, Retrieve all historical frames within `lookback_seconds` before the latest times, Hot-swap threshold rules and weights at runtime. (+5 more)

### Community 7 - "Directory Video Testing & Extraction"
Cohesion: 0.08
Nodes (25): 1. Prerequisites, 2. Create Virtual Environment, 3. Install Dependencies, 4. Install MMDetection via OpenMIM, 🥊 Advanced Violence Detection, Configure the Model Paths (PowerShell), Core Features, 🔥 Custom Object Detection (RTMDet) (+17 more)

### Community 8 - "diagnostic_run.py"
Cohesion: 0.19
Nodes (14): calculate_kinematics(), _cosine_similarity(), _fill_missing(), ndarray, Applies Exponential Moving Average (EMA) smoothing to a sequence     of 2D keypo, Computes 1st through 3rd order kinematic derivatives from a     time-series of 2, Replaces (0, 0) entries (undetected keypoints) with the last     known valid pos, Computes the cosine similarity between two 2D vectors.      Returns a value in [ (+6 more)

### Community 9 - "Fall Detection & Emergency Monitoring"
Cohesion: 0.08
Nodes (17): _compute_angle(), _extract_kp(), PersonHistoryManager, PersonTrackHistory, Any, ndarray, Update historical buffers with new frame detections and compute derived kinemati, Return a dictionary matching the exact schema expected by action_math.py. (+9 more)

### Community 11 - "video_websocket"
Cohesion: 0.20
Nodes (10): check_falls(), check_violence_between_pairs(), cleanup_stale_tracks(), Any, Iterates over all pairs of tracked persons. For each pair within     PROXIMITY_T, Checks every tracked person for fall events using the rule-based     MedicalEmer, Removes person_states entries that have been inactive for longer than timeout_se, Main WebSocket endpoint for real-time video processing.      Protocol:       1) (+2 more)

### Community 17 - ".draw_hud"
Cohesion: 0.25
Nodes (6): CrowdMetrics, Detections, ndarray, Render a compact, high-visibility Crowd Analytics HUD onto the frame., Immutable telemetry data emitted by CrowdCounter per frame., Update crowd metrics from current frame ByteTrack detections.          Args:

## Knowledge Gaps
- **25 isolated node(s):** `graphify`, `Workflow: graphify`, `🔥 Custom Object Detection (RTMDet)`, `🎯 Persistent Multi-Object Tracking (Pre-Pose ByteTrack Architecture)`, `👥 Real-Time Crowd Analytics` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PersonHistoryManager` connect `Fall Detection & Emergency Monitoring` to `Action Math & Violence Detection`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `MedicalEmergencyAnalyzer` connect `Action Math & Violence Detection` to `video_websocket`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `CrowdCounter` connect `Action Math & Violence Detection` to `.draw_hud`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **What connects `action_math.py — Production-Grade Kinematics Module for Violence & Fall Detectio`, `Applies Exponential Moving Average (EMA) smoothing to a sequence     of 2D keypo`, `Computes 1st through 3rd order kinematic derivatives from a     time-series of 2` to the rest of the system?**
  _104 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Vision Models & BBox Processing` be split into smaller, more focused modules?**
  _Cohesion score 0.11174242424242424 - nodes in this community are weakly interconnected._
- **Should `YOLO to COCO Conversion` be split into smaller, more focused modules?**
  _Cohesion score 0.12666666666666668 - nodes in this community are weakly interconnected._
- **Should `Action Math & Violence Detection` be split into smaller, more focused modules?**
  _Cohesion score 0.0859538784067086 - nodes in this community are weakly interconnected._