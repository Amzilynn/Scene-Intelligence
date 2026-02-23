# IA Camera Challenge - Computer Vision Pipeline

A state-of-the-art CV pipeline for security camera analysis featuring YOLOv26-s, real-time person tracking, pose estimation, and a hybrid Service Provider Identification layer.

## 🚀 Key Features

### 💎 Core Intelligence
- **YOLOv26-s Upgrade**: Leverages the latest YOLO26 architecture for ultra-efficient, end-to-end NMS-free human detection and pose estimation.
- **Hybrid Role Identification**: A multi-layered logic to distinguish Service Providers (Staff) from Visitors using:
    - **Visual Markers**: Scanning for registries, badges, and tablets on person crops.
    - **Social Centrality**: Analysis of interaction hub behavior (who interacts with the most unique people).
    - **Spatial Dispersion**: Tracking scene coverage patterns (patrolling vs. territorial stay).
- **Social Interaction (STAS)**: Geometry-based detection of behaviors like Talking, Service/Helping, Walking together, and Space VIOLATIONS.

### 🎥 Robust Pipeline
- **Multi-Object Tracking**: BoxMOT (DeepOCSORT) for persistent IDs and cross-scene re-identification.
- **Demographic Analysis**: MiVOLO (Face + Body) for precise age and gender estimation.
- **Emotion Analysis**: DeepFace for facial emotion and sentiment trend analysis.
- **JSONL Logging**: Structured, line-delimited JSON data for easy downstream analytics.

## 📁 Clean Repository Structure

```bash
IA-Camera-Challenge/
├── cv_pipeline/
│   ├── detection/          # YOLOv26-s & Provider Identification
│   ├── tracking/           # BoxMOT tracking module
│   ├── pose_estimation/    # RTMPose & YOLO-Pose
│   ├── emotion_analysis/   # DeepFace & MiVOLO
│   ├── social_interaction/ # Hybrid Role Inference & STAS
│   └── utils/              # JSON Scene Describer
├── scripts/
│   ├── run_full_pipeline.py  # Main entry point (MP4/JSON output)
│   └── download_model.py     # Setup helper
├── models/                 # Model weights (YOLO26-S, Face, MiVOLO)
├── final_output.mp4        # Annotated high-res video output
└── scene_log.json          # Structured frame-by-frame data
```

## 🛠 Quick Start

### Prerequisites
- **Python**: 3.11+
- **GPU**: NVIDIA GPU (GTX 1650+) with CUDA 12.1+

### Installation
1. **Setup Env**: `python -m venv venv` and activate it.
2. **PyTorch**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. **Requirements**: `pip install -r requirements.txt`
4. **Models**: Ensure `yolo26s.pt` is in the root or will be auto-downloaded by Ultralytics.

### Usage
```bash
python scripts/run_full_pipeline.py  # Default: vd2.mp4
```

## 📊 Outputs
- **`final_output.mp4`**: Annotated video with ID, Role, Emotion, and Interaction tags.
- **`scene_log.json`**: JSONL structured data for high-level scene understanding.

---
**Maintained by**: Amzilynn | **Engine**: YOLOv26-s + Custom Hybrid Logic
