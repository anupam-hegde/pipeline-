# Graph Report - pipeline  (2026-07-08)

## Corpus Check
- 19 files · ~61,288 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 237 nodes · 394 edges · 16 communities (11 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `431ffbfc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Vision Models & BBox Processing
- YOLOX Surveillance Configuration
- YOLO to COCO Conversion
- Action Math & Violence Detection
- FastAPI WebSocket & Webcam Server
- Dataset Preparation & Directory Creation
- Real-Time Tracking & State Management
- Directory Video Testing & Extraction
- Video Pose Extraction & Testing
- Fall Detection & Emergency Monitoring
- RTMPose Pipeline Inference
- Graphify AI Agent Rules
- Graphify Workflow Automation
- YOLOX Training Log History
- Project Python Dependencies

## God Nodes (most connected - your core abstractions)
1. `PersonHistoryManager` - 15 edges
2. `detect_advanced_violence()` - 14 edges
3. `PersonTrackHistory` - 12 edges
4. `match_keypoints_to_bbox()` - 12 edges
5. `detect_fall()` - 11 edges
6. `ObjectDetectionBatch` - 11 edges
7. `create_pose_pipeline()` - 11 edges
8. `create_object_detector()` - 11 edges
9. `extract_keypoint()` - 10 edges
10. `draw_object_detections()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `export_and_verify()` --calls--> `Exp`  [EXTRACTED]
  scripts/export_onnx.py → configs/yolox_surveillance.py
- `main()` --calls--> `Exp`  [EXTRACTED]
  scripts/train_model.py → configs/yolox_surveillance.py
- `check_violence_between_pairs()` --calls--> `detect_advanced_violence()`  [EXTRACTED]
  backend/main.py → backend/action_math.py
- `check_falls()` --calls--> `detect_fall()`  [EXTRACTED]
  backend/main.py → backend/action_math.py
- `load_models()` --calls--> `CrowdCounter`  [EXTRACTED]
  backend/main.py → backend/crowd_counter.py

## Import Cycles
- None detected.

## Communities (16 total, 5 thin omitted)

### Community 0 - "Vision Models & BBox Processing"
Cohesion: 0.12
Nodes (17): bbox_iou(), _decode_yolox_output(), _extract_bbox(), _extract_score(), _letterbox(), _nms(), _parse_pose_instances(), PoseDetectionBatch (+9 more)

### Community 1 - "YOLOX Surveillance Configuration"
Cohesion: 0.13
Nodes (16): BaseExp, Exp, yolox_surveillance.py — YOLOX-s Experiment Config for Surveillance Detection  Mo, Override to point images at merged_dataset/val/images/<file_name>, Override to point images at merged_dataset/train/images/<file_name>         YOLO, export_and_verify(), main(), Path (+8 more)

### Community 2 - "YOLO to COCO Conversion"
Cohesion: 0.18
Nodes (15): convert_dataset_root(), convert_split_to_coco(), get_image_info(), main(), Any, Path, convert_yolo_to_coco.py — Fast Parallel YOLO to COCO JSON Annotation Converter, Converts standard train/val splits under dataset_root to COCO JSONs. (+7 more)

### Community 3 - "Action Math & Violence Detection"
Cohesion: 0.10
Nodes (39): calculate_kinematics(), _cosine_similarity(), detect_advanced_violence(), detect_fall(), extract_keypoint(), _fill_missing(), ndarray, action_math.py — Advanced Kinematics Module for Violence Detection  Implements 3 (+31 more)

### Community 4 - "FastAPI WebSocket & Webcam Server"
Cohesion: 0.12
Nodes (17): CrowdDensityEstimator, DensityLevel, DensityMetrics, DensityThresholds, Detections, ndarray, Crowd density classification levels., Render a color-coded Crowd Density badge onto the video stream. (+9 more)

### Community 5 - "Dataset Preparation & Directory Creation"
Cohesion: 0.36
Nodes (8): create_directories(), generate_yaml(), main(), process_dataset(), Path, Generates the data.yaml configuration file for YOLO training., Creates the YOLO directory structure inside OUTPUT_DIR., Scans a source dataset (either direct images/labels or split train/val/test fold

### Community 6 - "Real-Time Tracking & State Management"
Cohesion: 0.22
Nodes (9): check_falls(), check_violence_between_pairs(), cleanup_stale_tracks(), Iterates over all pairs of tracked persons. For each pair within     PROXIMITY_T, Checks every tracked person for a fall event using the     head velocity + torso, Removes person_states entries that have been inactive for longer than timeout_se, Main WebSocket endpoint for real-time video processing.      Protocol:       1), video_websocket() (+1 more)

### Community 7 - "Directory Video Testing & Extraction"
Cohesion: 0.08
Nodes (23): 1. Prerequisites, 2. Create Virtual Environment, 3. Install Dependencies, 4. Install MMDetection via OpenMIM, 🥊 Advanced Violence Detection, Configure the Model Paths (PowerShell), Core Features, 🔥 Custom Object Detection (RTMDet) (+15 more)

### Community 8 - "Video Pose Extraction & Testing"
Cohesion: 0.16
Nodes (9): CrowdCounter, CrowdMetrics, Detections, ndarray, Render a high-visibility Crowd Analytics HUD onto the frame., Immutable telemetry data emitted by CrowdCounter per frame., Real-time crowd occupancy and visitor analytics module.      Consumes ByteTrack, Reset all occupancy counters and historical telemetry. (+1 more)

### Community 9 - "Fall Detection & Emergency Monitoring"
Cohesion: 0.08
Nodes (17): _compute_angle(), _extract_kp(), PersonHistoryManager, PersonTrackHistory, Any, ndarray, Update historical buffers with new frame detections and compute derived kinemati, Return a dictionary matching the exact schema expected by action_math.py. (+9 more)

## Knowledge Gaps
- **23 isolated node(s):** `graphify`, `Workflow: graphify`, `🔥 Custom Object Detection (RTMDet)`, `🚑 Robust Medical Emergency (Fall) Detection`, `🥊 Advanced Violence Detection` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PersonHistoryManager` connect `Fall Detection & Emergency Monitoring` to `Action Math & Violence Detection`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `CrowdCounter` connect `Video Pose Extraction & Testing` to `Action Math & Violence Detection`, `FastAPI WebSocket & Webcam Server`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `CrowdDensityEstimator` connect `FastAPI WebSocket & Webcam Server` to `Action Math & Violence Detection`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **What connects `action_math.py — Advanced Kinematics Module for Violence Detection  Implements 3`, `Applies Exponential Moving Average (EMA) smoothing to a sequence     of 2D keypo`, `Computes 1st through 3rd order kinematic derivatives from a     time-series of 2` to the rest of the system?**
  _87 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Vision Models & BBox Processing` be split into smaller, more focused modules?**
  _Cohesion score 0.12169312169312169 - nodes in this community are weakly interconnected._
- **Should `YOLOX Surveillance Configuration` be split into smaller, more focused modules?**
  _Cohesion score 0.12648221343873517 - nodes in this community are weakly interconnected._
- **Should `Action Math & Violence Detection` be split into smaller, more focused modules?**
  _Cohesion score 0.10048309178743961 - nodes in this community are weakly interconnected._