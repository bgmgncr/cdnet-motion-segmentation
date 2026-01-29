# Motion Segmentation for Video Surveillance

A comprehensive motion segmentation system designed to accurately detect and isolate moving objects in video sequences, with specialized algorithms tailored for challenging scenarios like shadows, dynamic backgrounds, and camera movement.

## Project Overview

This project implements multiple motion segmentation algorithms optimized for different video conditions found in the ChangeDetection.NET (CDnet) benchmark dataset. Each algorithm is carefully tuned to handle specific challenges:

- **Baseline/Standard Scenes**: MOG2 background subtraction
- **Shadow Scenarios**: Color-space aware shadow detection
- **Dynamic Backgrounds**: Temporal median + optical flow hybrid approach
- **Pan-Tilt-Zoom (PTZ)**: MOG2 with camera motion handling
- **Bad Weather**: MOG2 with weather-resilient parameters

## Dataset Structure

The project uses the CDnet benchmark dataset organized by scenario type:

```
dataset/
├── baseline/           # Standard indoor/outdoor scenes
│   ├── highway, office, pedestrians, PETS2006
├── badWeather/        # Challenging weather conditions
│   ├── blizzard, skating, snowFall, wetSnow
├── dynamicBackground/ # Moving background elements
│   ├── boats, canoe, fall, fountain01, fountain02, overpass
├── PTZ/              # Pan-Tilt-Zoom camera movement
│   ├── continuousPan, intermittentPan, twoPositionPTZCam, zoomInZoomOut
└── shadow/           # Shadow scenarios
    ├── backdoor, bungalows, busStation, copyMachine, cubicle, peopleInShade
```

Each sequence contains:
- `input/`: Video frames (PNG/JPG)
- `groundtruth/`: Ground truth foreground masks
- `temporalROI.txt`: Temporal Region of Interest (valid frame ranges)

## Core Components

### 1. **Pipeline** (`pipeline.py`)

The main processing orchestrator that selects the appropriate algorithm based on sequence category:

- **Shadow Sequences**: Uses `ShadowAwareBackgroundSubtractor` with HSV/YUV color space analysis
- **Dynamic Background Sequences**: Uses `HybridDynamicBackgroundSubtractor` combining temporal median + optical flow
- **Other Sequences**: Uses standard MOG2 background subtraction

**Key Features:**
- Automatic algorithm selection based on input category
- Post-processing with morphological operations (opening, closing)
- Small blob removal (area threshold: 300 pixels)
- Overlay generation for visual verification

### 2. **Shadow-Aware Algorithm** (`shadow_algorithm.py`)

Specialized algorithm for accurately detecting motion while reducing shadow artifacts.

**Technical Approach:**
- **Color Space Analysis**: Uses BGR, HSV, and YUV simultaneously
- **HSV Shadow Detection**:
  - Low saturation (< 30) indicates reduced color variation typical of shadows
  - Value range 20-100 captures darkened regions without pure black
- **YUV Shadow Detection**:
  - Reduced brightness (Y < 100) characteristic of shadows
  - Normal chrominance components (U, V near 128) distinguish shadows from true objects
- **Multi-Score Combination**: Weighted combination of HSV and YUV scores
- **Adaptive Learning**: More permissive in early frames (< 50 frames) while MOG2 is learning

**Key Features:**
- `ShadowAwareBackgroundSubtractor` class with configurable history and variance threshold
- Frame-by-frame shadow mapping for diagnosis
- Morphological filtering to remove noise

### 3. **Hybrid Dynamic Background Algorithm** (`dynamic_background_algorithm.py`)

Robust algorithm for sequences with moving background elements like water, vegetation, or reflections.

**Technical Approach:**
- **Temporal Median Modeling**:
  - Maintains sliding window of recent frames (default: 15 frames)
  - Calculates median pixel values across the window
  - Naturally models slow-moving backgrounds
  - More robust to outliers than mean-based approaches

- **Optical Flow Analysis**:
  - Detects pixel-level motion using Farneback algorithm
  - Compares current frame optical flow against expected background flow
  - Identifies moving foreground objects that deviate from background motion patterns

- **Hybrid Fusion**:
  - Combines temporal median mask and optical flow detection
  - Requires agreement between both methods to reduce false positives
  - Adaptive thresholding based on confidence metrics

**Key Features:**
- `HybridDynamicBackgroundSubtractor` class
- Configurable window size, flow threshold, and area threshold
- Handles reflections and water motion effectively

### 4. **Optical Flow Algorithm** (`optical_flow_algorithm.py`)

Pure optical flow-based motion detection for comparison purposes.

**Technical Approach:**
- Uses Farneback algorithm for dense optical flow computation
- Detects motion based solely on optical flow magnitude
- Simpler and faster than the hybrid approach
- Useful for algorithm comparison on dynamic backgrounds

**Key Features:**
- `OpticalFlowBackgroundSubtractor` class
- Configurable flow magnitude threshold
- Motion consistency checking across frames

### 5. **Dataset I/O** (`dataset_io.py`)

Utilities for accessing CDnet dataset structure.

**Functions:**
- `list_categories()`: Returns all available scenario types
- `list_sequences(category)`: Returns sequences in a given category
- `list_frames(category, sequence)`: Returns sorted frame paths for a sequence

## Usage

### Basic Processing

Process a single sequence with automatic algorithm selection:

```bash
python run.py --category baseline --sequence highway --max_frames 300
```

Process all frames in a sequence:

```bash
python run.py --category dynamicBackground --sequence fountain01 --max_frames -1
```

List available categories:

```bash
python run.py --list
```

List sequences in a category:

```bash
python run.py --list_sequences --category shadow
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--category` | `baseline` | Scenario category (baseline, badWeather, dynamicBackground, PTZ, shadow) |
| `--sequence` | First in category | Specific sequence name |
| `--max_frames` | 300 | Maximum frames to process (-1 for all) |
| `--out_dir` | `outputs` | Output directory root |
| `--list` | - | List all categories and exit |
| `--list_sequences` | - | List sequences in category and exit |

### Algorithm Comparison

Compare hybrid vs optical flow only approaches on dynamic backgrounds:

```bash
python compare_algorithms.py --category dynamicBackground --sequence fountain01 --algorithm both
```

Options:
- `--algorithm hybrid`: Only hybrid algorithm
- `--algorithm optical_flow`: Only optical flow
- `--algorithm both`: Compare both (default)

### Visualization

View segmentation results with overlays:

```bash
python view_outputs.py --category dynamicBackground --sequence fountain01 --max_frames 150
```

## Output Structure

Processed results are saved to `outputs/` (or custom output directory):

```
outputs/
├── {category}/
│   └── {sequence}/
│       ├── masks/           # Binary foreground masks (PNG)
│       └── overlays/        # Frame overlays with detected foreground (PNG)
```

- **Masks**: White (255) = foreground, Black (0) = background
- **Overlays**: Original frame with transparent red overlay on detected moving objects

## Algorithm Selection Rationale

### MOG2 (Mixture of Gaussians)
- **When**: Baseline, bad weather, PTZ
- **Why**: Well-established, efficient, handles gradual illumination changes
- **Limitations**: Struggles with shadows, dynamic backgrounds

### Shadow-Aware Algorithm
- **When**: Shadow scenarios
- **Why**: Multi-color space analysis specifically designed to distinguish shadows from objects
- **Advantages**: Reduces false positives from shadow movement
- **Trade-offs**: Slower than MOG2, requires learning phase

### Hybrid (Temporal Median + Optical Flow)
- **When**: Dynamic backgrounds (water, vegetation, reflections)
- **Why**: Temporal median adapts to slow background motion; optical flow detects true foreground motion
- **Advantages**: Handles moving backgrounds naturally
- **Trade-offs**: Computationally more expensive, requires tuning window size

## Key Parameters & Tuning

### MOG2
- `history=500`: Number of frames to keep in model
- `varThreshold=16`: Variance threshold for background model
- `detectShadows=True`: Enable shadow detection

### Shadow Algorithm
- `history=300`: Frames in MOG2 model
- `var_threshold=20`: MOG2 variance threshold
- HSV thresholds: Saturation < 30, Value 20-100
- YUV thresholds: Y < 100, U/V deviation < 20

### Hybrid Algorithm
- `window_size=15`: Frames in temporal median buffer
- `flow_threshold=0.5`: Optical flow magnitude threshold
- `area_threshold=300`: Minimum blob area (pixels)

### Morphological Filtering
- Kernel: 3×3 elliptical
- Opening: 1 iteration (removes noise)
- Closing: 2 iterations (fills small holes)

## Technical Decisions

1. **Color Space Selection**: HSV for perceptual similarity, YUV for lighting-independent analysis
2. **Temporal Median over Mean**: More robust to outliers and sudden scene changes
3. **Morphological Operations**: Essential for removing noise and connecting broken regions
4. **Multi-Algorithm Approach**: Different scenarios require different methods
5. **Blob Area Filtering**: Removes noise while preserving real foreground objects

## Output Examples

### Baseline (MOG2)
- Accurate foreground detection
- Clean boundaries
- Minimal noise

### Shadow Scenarios
- Reduced shadow artifacts
- Improved object boundary accuracy
- Better handling of varying shadow intensity

### Dynamic Backgrounds
- Stable foreground detection despite background motion
- Distinguishes between background flow and object motion
- Handles water, vegetation, and reflections

## Dependencies

- **OpenCV** (`cv2`): Core computer vision algorithms
- **NumPy**: Numerical operations
- **tqdm**: Progress bar visualization

## Performance Characteristics

| Algorithm | Speed | Accuracy (shadows) | Accuracy (dynamic BG) | Memory |
|-----------|-------|-------------------|----------------------|--------|
| MOG2 | Fast | Low | Medium | Low |
| Shadow-Aware | Medium | High | Medium | Medium |
| Hybrid | Medium | Medium | High | Medium |
| Optical Flow | Slow | Low | Medium | High |

## Future Improvements

- [ ] Deep learning-based segmentation (U-Net, Mask R-CNN)
- [ ] Adaptive parameter tuning per sequence
- [ ] GPU acceleration for optical flow
- [ ] Real-time processing optimization
- [ ] Evaluation metrics computation and reporting
- [ ] Confidence maps and uncertainty estimation
