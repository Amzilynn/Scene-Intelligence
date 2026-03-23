# IA Camera Challenge - Scene Intelligence Pipeline

A sophisticated, high-performance computer vision pipeline designed for human-centric scene understanding and interaction analysis. This system integrates state-of-the-art deep learning models to provide real-time insights into human behavior, social dynamics, and service efficiency within complex environments.

## Core Intelligence Modules

### REACT++ Scene Graph Generation
The engine leverages REACT++ for advanced Scene Graph Generation (SGG). Beyond simple object detection, it maps relationships between humans and their environment, enabling a semantic understanding of the scene. It identifies triplets consisting of subjects, predicates, and objects to build a dynamic knowledge graph of every frame.

### Spatial-Temporal Interaction Mode (STAS)
The STAS module performs continuous analysis of human-to-human relationships. Key features include:
- Synchronicity Analysis: Cross-correlation of movement velocity and head orientation to identify group bonding.
- Personal Space Monitoring: Constructing dynamic polygons around tracked individuals to detect Intimate Space violations.
- Interaction Smoothing: A temporal buffer system that uses majority-voting to eliminate semantic flickering and ensure stable relationship labels.

### Specialized Human Analysis
- HSEmotion Analysis: Utilizes state-of-the-art emotion detection models to track real-time satisfaction levels and sentiment trends.
- MiVOLO Demographics: Provides high-precision age and gender estimation using multi-modal face and body feature extraction.
- RTMPose Estimation: High-frequency pose tracking used for posture detection (Standing, Sitting, Crouching) and gesture analysis.

## Advanced Behavioral Insights

### Unsupervised Role Discovery
The pipeline automatically distinguishes between different roles (e.g., Staff vs. Visitors) without manual labeling. This is achieved through an multi-factor algorithm analyzing:
- Spatial Centrality: Persistence at specific service posts or counters.
- Mobility Index: Analysis of total distance covered relative to time elapsed.
- Hand-Object Interaction (HOI): Tracking interactions with specific equipment such as cashier registries or office hardware.

### Intent & Security Analytics
The system detects specific behavioral patterns to generate actionable insights:
- Scanning Behavior: Identifies stationary individuals with high-variance head orientation, signaling a need for pre-emptive service.
- Theft Heuristics: Monitors hand occlusion and orientation relative to high-value objects and staff presence.

## Cinematic HUD Visualization
The visualization engine features a professional-grade HUD with:
- Glassmorphism Interface: Semi-transparent, blur-effect intelligence cards for role and satisfaction metrics.
- Glowing Energy Links: Curved Bezier pulses that visualize social dynamics and interaction types between individuals.
- Dynamic Dashboard: A real-time summary of total counts, staff-to-visitor ratios, active engagements, and the Customer Satisfaction Index.

## Repository Structure

```text
IA-Camera-Challenge/
├── cv_pipeline/
│   ├── detection/          # REACT++ SGG and human detection
│   ├── tracking/           # DEEPOCSORT multi-object tracking
│   ├── emotion_analysis/   # HSEmotion sentiment tracking
│   ├── social_interaction/ # STAS relationship engine
│   ├── pose_estimation/    # RTMPose analysis
│   └── utils/              # Cinematic HUD and scene logs
├── SGG-Benchmark/          # Core SGG weights and utilities
├── models/                 # Pre-trained deep learning weights
└── scripts/                # Entry points for batch and real-time processing
```

## Setup and Deployment

### Requirements
- Python 3.11 or higher
- NVIDIA GPU with CUDA 12.1 support
- Minimum 16GB System RAM

### Installation
1. Initialize the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the necessary model weights as specified in the model configuration files.

### Execution
Run the full intelligence pipeline on a video file or live stream:
```bash
python cv_pipeline/scripts/run_full_pipeline.py [video_path] --output [output_name.mp4]
```

## Data Output
The system generates a high-fidelity intelligence log in JSONL format, documenting every detected entity, relationship, and behavioral event with precise timestamps for downstream analytical processing.

---
**Maintained by**: Amzilynn | **Engine**: REACT++ SGG + Custom Scene Intelligence
