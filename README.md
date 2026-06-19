# Real-Time Surveillance & Medical Emergency Pipeline

A high-performance computer vision pipeline built with **YOLOv11**, **ByteTrack**, and **OpenCV**. This system is designed for real-time human tracking and advanced behavioral analysis, specifically engineered to eliminate false positives in complex real-world environments.

## Core Features

### 🚑 Robust Medical Emergency (Fall) Detection
Standard fall detection relies on hardcoded pixel drops, which fail due to camera perspective. This pipeline uses a **Multi-Factor Scale-Invariant Heuristic**:
- **Scale-Invariant Drop:** Calculates head velocity relative to the person's own bounding box height, ensuring flawless detection whether the person is 5 feet or 50 feet away.
- **Floor Test (Aspect Ratio):** Continuously monitors bounding box aspect ratio to determine if a person is lying horizontally.
- **Crumple & Slump Detection:** Heart attacks or fainting often result in vertical crumpling rather than flat falls. The system tracks the vertical distance between the head and hips to detect slumping.
- **Debounce State Machine:** A 10-frame debounce prevents false positives from rapid movements like tying shoes.

### 🥊 Advanced Violence & Scuffle Detection
Uses 3rd-order kinematics to distinguish genuine violent impulses from periodic exercise or waving.
- **Jerk Spike Detection:** Computes the 3rd derivative of wrist motion to isolate sudden impulses (punches/strikes).
- **Momentum Transfer:** Calculates the cosine similarity (directional alignment) between the attacker's wrist velocity and the victim's head velocity to detect physical recoil.

### 🔥 Custom Object Detection
Includes parallel inference for a custom-trained object detection model to flag:
- Fire / Smoke
- Weapons (Knives, Firearms)

### 🌐 FastAPI WebSocket Backend
Includes a `main.py` FastAPI server that streams annotated video frames and telemetry alerts over WebSockets, allowing easy integration with React/Next.js frontend dashboards.

---

## 🚀 Installation & Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/surveillance-pipeline.git
   cd surveillance-pipeline
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv11 Pose Model**:
   Ensure you have the base pose model in the `models/` directory:
   ```bash
   mkdir models
   # Download yolo11n-pose.pt or place it in the models directory
   ```

---

## 💻 Usage

The pipeline includes several entry points depending on your use case:

### 1. Live Webcam Testing
Run the standalone webcam script to test the violence and fall detection logic in real time on your own camera.
```bash
python backend/test_webcam.py
```
*(Press `q` to quit the window)*

### 2. Offline Video Processing
Process a pre-recorded CCTV or test video file.
```bash
python backend/test_video.py --input path/to/your/video.mp4 --output output.mp4
```

### 3. Start the Backend Server
Launch the FastAPI WebSocket server for production integration.
```bash
python backend/main.py
```
The server will boot up at `ws://localhost:8000/ws/video` and `http://localhost:8000/health`.

---

## 🛠 Project Structure

- `backend/action_math.py`: The brains. Contains all kinematic math, EMA smoothing, and heuristic logic.
- `backend/main.py`: FastAPI server for streaming data.
- `backend/test_webcam.py`: Live debugging and visualizer script.
- `backend/test_video.py`: Offline video rendering script.
- `scripts/`: Helper scripts for repairing labels and training custom models.
