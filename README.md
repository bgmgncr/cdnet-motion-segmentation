# Motion Segmentation with Classical Image Processing (CDNet 2014)

This project explores motion-based foreground segmentation using classical image processing techniques. The goal is to detect moving regions in video sequences while suppressing static background regions.

## Dataset
The project uses the **CDNet 2014 dataset**, which contains multiple categories representing real-world challenges such as:
- baseline scenes
- shadow-heavy scenes
- dynamic background scenes (e.g., fountains, water)

## Method
- Background subtraction using **MOG2**
- Morphological operations for noise removal
- Median filtering for smoother masks
- HSV/YUV color space analysis for shadow suppression
- Qualitative evaluation using overlay images

## Experiments
Representative sequences were tested from:
- baseline category
- shadow category (e.g., copyMachine)
- dynamic background category (e.g., fountain01)

## Key Observations
- Motion detection works well in stable scenes
- Shadows cause false detections due to illumination changes
- Dynamic backgrounds produce flickering and incomplete masks
- Motion-based methods detect motion, not object identity

## Technologies
- Python
- OpenCV
- NumPy

## Notes
This project focuses on understanding the behavior and limitations of classical background subtraction methods rather than achieving perfect segmentation.
