# IA Camera Challenge - Scene Intelligence Pipeline

A sophisticated, high-performance computer vision pipeline designed for human-centric scene understanding and interaction analysis. This system integrates state-of-the-art deep learning models to provide real-time insights into human behavior, social dynamics, and service efficiency within complex environments.

## Core Intelligence Modules

### REACT++ Scene Graph Generation
The engine leverages REACT++ for advanced Scene Graph Generation (SGG). Beyond simple object detection, it maps relationships between humans and their environment, enabling a semantic understanding of the scene. It identifies triplets consisting of subjects, predicates, and objects to build a dynamic knowledge graph of every frame.

### STAS: Spatial-Temporal Interaction Mode
The STAS module performs continuous analysis of human-to-human relationships, moving beyond frame-by-frame detection into temporal behavioral patterns.
- **Synchronicity Analysis**: Uses cross-correlation of movement velocity and head orientation to identify group bonding and mirrored behaviors.
- **Personal Space Micro-Polygons**: Constructs dynamic "bubbles" around tracked individuals using 17-point pose keypoints to detect intimate space violations and social comfort zones.
- **Intentional Focus (Ray Casting)**: Implements vector-based ray casting from a person's nose-to-neck axis to detect active engagement and "gaze intersection" with others.

### Specialized Human Analysis
- **HSEmotion Analysis**: State-of-the-art emotion detection models track real-time satisfaction levels and sentiment trends.
- **MiVOLO Demographics**: High-precision age and gender estimation using multi-modal face and body feature extraction.
- **RTMPose Estimation**: High-frequency pose tracking used for posture detection (Standing, Sitting, Crouching) and complex gesture analysis.

## Advanced Behavioral Insights

### Unsupervised Role Discovery (V2)
The pipeline automatically distinguishes between different roles (e.g., Staff vs. Visitors) using a multi-factor behavioral algorithm:
- **Spatial Centrality & Mobility Index**: Analyzes persistence at specific service posts vs. general movement patterns.
- **Social Reach**: Tracks the number of unique interactions and engagements per individual.
- **Hand-Object Interaction (HOI)**: Detects precise interactions with business-critical equipment like registries, laptops, and service hardware.

### Intent & Service Analytics
The system detects specific behavioral patterns to generate actionable alerts:
- **Pre-emptive Service (Scanning)**: Identifies stationary individuals with high-variance head movements, signaling they are looking for assistance before they even ask.
- **Security Heuristics**: Monitors hand occlusion and orientation relative to high-value objects and staff presence to flag unusual behavior.

## Cinematic HUD Visualization
The visualization engine features a professional-grade, "Cyber-Agent" HUD designed for clarity and impact:
- **Glassmorphism Interface**: Semi-transparent, Gaussian-blurred intelligence cards for role and satisfaction metrics, providing a true frosted glass effect.
- **Glowing Energy Links**: Adaptive Bezier pulses that visualize social dynamics and interaction types (Talking, Service, Group Bond) between individuals.
- **Dynamic System Vibe Dashboard**: A real-time summary of the "Vibe Analysis" (Status: Nominal/Critical), staff-to-visitor ratios, and the aggregate Customer Satisfaction Index.
- **Interactive Scanlines & Edge Glow**: Cinematic overlays that react to the scene's overall satisfaction and activity levels.

## Repository Structure

```text
IA-Camera-Challenge/
├── cv_pipeline/
│   ├── detection/          # REACT++ SGG and human detection
│   ├── tracking/           # DEEPOCSORT multi-object tracking
│   ├── emotion_analysis/   # HSEmotion sentiment tracking
│   ├── social_interaction/ # STAS relationship engine & Role Discovery
│   ├── pose_estimation/    # RTMPose analysis
│   └── utils/              # UIProcessor (Glassmorphism HUD) & Scene Describer
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
The system generates a high-fidelity intelligence log in JSONL format, documenting every detected entity, relationship, and behavioral event with precise temporal markers for downstream analytical processing.

---
**Maintained by**: Amzilynn | **Engine**: REACT++ SGG + Custom Scene Intelligence
