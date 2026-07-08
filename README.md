# Real-Time Surveillance & Object Detection Pipeline

A high-performance computer vision pipeline built with **RTMPose**, **RTMDet** (OpenMMLab, Apache-2.0), **ByteTrack**, and **OpenCV**. Engineered for real-time CCTV inference — detecting **fire**, **weapons**, and analyzing human behaviour for fall and violence events.

---

## Core Features

### 🔥 Custom Object Detection (RTMDet)
- Trained on a custom merged dataset of **80,406 images** (64,339 train / 16,067 val).
- Detects: **Fire/Smoke** and **Weapons (knives, firearms)**.
- Apache-2.0 licensed OpenMMLab RTMDet-tiny — optimized for real-time CCTV at 300+ FPS on GPU.

### 🎯 Persistent Multi-Object Tracking (Pre-Pose ByteTrack Architecture)
- **Track-Aware Top-Down Pose Inference:** Integrates ByteTrack immediately after object detection. MMPose crops directly around tracked individuals, maintaining consistent identities across frames.
- **Memory-Optimized Person History Manager (`history_manager.py`):** Bounded circular ring buffers ($O(1)$ memory per track) store bounding boxes, 17 COCO keypoints, biomechanical joint angles (`elbows`, `knees`, `hips`), kinematic velocities (`vx, vy`), and timestamps with automatic timeout eviction for 24/7 CCTV stability.

### 👥 Real-Time Crowd Analytics
- **Unique Visitor & Active Headcount (`crowd_counter.py`):** Tracks current active occupants, peak headcount, and cumulative unique visitors using set-based hashing with temporal stabilization debounce.
- **Occlusion-Aware Crowd Density Estimation (`crowd_density.py`):** Computes true occupied frame area using a downscaled boolean numpy grid mask to avoid double-counting overlapping bounding boxes. Classifies scene density into **LOW**, **MEDIUM**, or **HIGH** with visual HUD overlays.

### 🚑 Robust Medical Emergency (Fall) Detection
- **Scale-Invariant Drop:** Head velocity relative to bounding box height — works at any camera depth.
- **Floor Test (Aspect Ratio):** Monitors bounding box shape to detect horizontal body position.
- **Crumple Detection:** Tracks head-to-hip vertical distance to catch silent collapses and fainting.
- **Debounce State Machine:** 10-frame latch prevents false positives from normal crouching.

### 🥊 Advanced Violence Detection
- **Jerk Spike Detection:** 3rd-order derivative of wrist motion to isolate genuine strikes.
- **Momentum Transfer:** Cosine similarity between attacker wrist velocity and victim head velocity.

### 🌐 FastAPI WebSocket Backend
Streams annotated video frames, crowd telemetry, and safety alerts over WebSocket, ready for React/Next.js frontends.

---

## 🛠 Installation & Setup

### 1. Prerequisites
Install PyTorch with CUDA support first (required before any pip installs):

```bash
# For CUDA 12.1 (RTX 3050 / RTX 40-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available:
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install MMDetection via OpenMIM
```bash
mim install mmdet mmpose
```

---

## 🏋️ Training the Custom Object Detection Model (RTMDet)

### Step 1: Prepare the Dataset
Consolidates all source datasets (fire/smoke, weapons) into the merged YOLO-format dataset:
```bash
python scripts/prepare_dataset.py
```

### Step 2: Run the Complete Training Pipeline
This single command will:
1. Convert YOLO annotations → COCO JSON (required by MMDetection).
2. Generate an optimized RTMDet-tiny config (FP16 AMP, 6 GB VRAM tuned).
3. Launch CUDA-accelerated training for 100 epochs.
4. Export the best checkpoint to ONNX and verify inference.

```bash
python scripts/train_model.py
```

> **Hardware Note:** Requires an NVIDIA GPU with ≥6 GB VRAM. Tested on RTX 3050 Laptop (6 GB).
> Mixed precision (FP16) is automatically enabled to fit within VRAM budget.

**Outputs:**
| File | Description |
|---|---|
| `models/surveillance_run/epoch_best.pth` | Best PyTorch checkpoint |
| `models/surveillance_run/rtmdet_surveillance.onnx` | ONNX model for production inference |
| `configs/rtmdet_surveillance.py` | Generated MMDetection configuration |
| `merged_dataset/annotations/` | COCO JSON annotation files |

### Step 3: Export ONNX Manually (Optional)
```bash
python scripts/export_onnx.py \
  --config configs/rtmdet_surveillance.py \
  --checkpoint models/surveillance_run/epoch_best.pth \
  --output models/surveillance_run/rtmdet_surveillance.onnx
```

---

## 💻 Running the Pipeline

### Configure the Model Paths (PowerShell)
```powershell
$env:RTMDET_CONFIG  = "configs/rtmdet_surveillance.py"
$env:RTMDET_WEIGHTS = "models/surveillance_run/epoch_best.pth"
$env:CV_DEVICE      = "cuda:0"
```

### Live Webcam Test
```bash
python backend/test_webcam.py
```
*(Press `q` to quit)*

### Offline Video Processing
```bash
python backend/test_video.py --input path/to/video.mp4 --output output.mp4
```

### Directory of Videos
```bash
python backend/test_video_dir.py --input-dir path/to/videos/ --output-dir output/
```

### Start FastAPI WebSocket Server
```bash
python backend/main.py
# WebSocket:    ws://localhost:8000/ws/video
# Health Check: http://localhost:8000/health
```

---

## 🗂 Project Structure

```
pipeline/
├── backend/
│   ├── vision_models.py      # RTMDet & RTMPose inference adapters (OpenMMLab, Apache-2.0) + ByteTrack
│   ├── history_manager.py    # Temporal telemetry manager (bboxes, keypoints, joint angles, velocities)
│   ├── crowd_counter.py      # Real-time active headcount & unique visitor counter
│   ├── crowd_density.py      # Occlusion-aware crowd density estimator & spatial occupancy
│   ├── action_math.py        # Kinematic math: jerk, momentum transfer, fall heuristics
│   ├── main.py               # FastAPI WebSocket server
│   ├── test_webcam.py        # Live webcam debugging
│   ├── test_video.py         # Offline video processing
│   └── test_video_dir.py     # Batch video directory processing
│
├── scripts/
│   ├── prepare_dataset.py    # Consolidates source datasets into merged_dataset/
│   ├── convert_yolo_to_coco.py  # Fast parallel YOLO → COCO JSON converter
│   ├── train_model.py        # Full RTMDet training pipeline (COCO convert → train → export)
│   ├── export_onnx.py        # RTMDet → ONNX export and ONNX Runtime verification
│   └── repair_labels.py      # Repairs malformed YOLO label files in merged dataset
│
├── configs/
│   └── rtmdet_surveillance.py   # Auto-generated MMDetection config (created by train_model.py)
│
├── merged_dataset/
│   ├── train/images/         # Training images
│   ├── train/labels/         # Training YOLO labels
│   ├── val/images/           # Validation images
│   ├── val/labels/           # Validation YOLO labels
│   ├── annotations/          # COCO JSON annotations (created by train_model.py)
│   └── data.yaml             # Dataset metadata
│
├── models/
│   └── surveillance_run/     # Training outputs (checkpoints, logs, ONNX)
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🔒 License

All core ML frameworks used are **Apache-2.0** licensed:
- [MMDetection](https://github.com/open-mmlab/mmdetection) — Apache-2.0
- [MMPose](https://github.com/open-mmlab/mmpose) — Apache-2.0
- [MMEngine](https://github.com/open-mmlab/mmengine) — Apache-2.0

Safe for commercial production deployment.
