"""
Hybrid motion segmentation algorithm for dynamic backgrounds.

Combines temporal median background modeling with optical flow analysis
to detect moving objects in sequences with dynamic backgrounds
(water, trees, reflections, etc.).
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from tqdm import tqdm


class HybridDynamicBackgroundSubtractor:
    """
    Hybrid background subtraction for dynamic backgrounds.
    
    Combines:
    1. Temporal Median: Uses median of recent frames as background model
    2. Optical Flow: Detects motion inconsistency using optical flow
    
    This handles moving backgrounds (water, trees) while still detecting
    true foreground motion (people, vehicles).
    """
    
    def __init__(self, window_size=15, flow_threshold=0.5, area_threshold=300):
        """
        Args:
            window_size: Number of frames to keep for median calculation
            flow_threshold: Optical flow magnitude threshold for motion detection
            area_threshold: Minimum blob size to keep
        """
        self.window_size = window_size
        self.flow_threshold = flow_threshold
        self.area_threshold = area_threshold
        
        # Frame buffer for temporal median
        self.frame_buffer = deque(maxlen=window_size)
        
        # Previous frame for optical flow calculation
        self.prev_frame_gray = None
        self.frame_count = 0
    
    def _get_temporal_median(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Calculate background as temporal median of recent frames.
        
        Args:
            current_frame: Current BGR frame
            
        Returns:
            Background estimate (BGR)
        """
        self.frame_buffer.append(current_frame)
        
        if len(self.frame_buffer) < 3:
            return current_frame.copy()
        
        # Stack frames and compute median along time axis
        frames_array = np.array(list(self.frame_buffer))
        background = np.median(frames_array, axis=0).astype(np.uint8)
        
        return background
    
    def _calculate_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow magnitude for motion detection.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Optical flow magnitude map
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = frame_gray.copy()
            return np.zeros_like(frame_gray, dtype=np.float32)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame_gray,
            frame_gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            15,    # winsize
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0      # flags
        )
        
        self.prev_frame_gray = frame_gray.copy()
        
        # Calculate magnitude of flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return magnitude
    
    def _refine_with_optical_flow(self, mask: np.ndarray, flow_magnitude: np.ndarray) -> np.ndarray:
        """
        Refine foreground mask using optical flow information.
        
        High flow regions are more likely to be foreground.
        Low flow regions are likely dynamic background.
        
        Args:
            mask: Initial foreground mask from background subtraction
            flow_magnitude: Optical flow magnitude map
            
        Returns:
            Refined foreground mask
        """
        # Normalize flow magnitude to 0-1
        flow_norm = np.clip(flow_magnitude / (self.flow_threshold * 2), 0, 1)
        
        # Create flow-based foreground likelihood
        # High flow = more likely foreground
        flow_mask = (flow_magnitude > self.flow_threshold).astype(np.uint8) * 255
        
        # Combine: keep pixels that are either in initial mask OR have high flow
        # But prioritize initial mask to avoid noise
        refined = cv2.bitwise_or(mask, flow_mask)
        
        return refined
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply hybrid background subtraction.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask (0=background, 255=foreground)
        """
        self.frame_count += 1
        
        # Get temporal median background
        background = self._get_temporal_median(frame)
        
        # Calculate difference between current frame and background
        # Use HSV for more robust color comparison
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Calculate euclidean distance in HSV space
        diff = np.sqrt(np.sum((frame_hsv - bg_hsv) ** 2, axis=2))
        
        # Threshold to get initial foreground
        # More permissive in early frames
        if self.frame_count < self.window_size:
            threshold = 20
        else:
            threshold = 15
        
        initial_mask = (diff > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Get optical flow magnitude
        flow_magnitude = self._calculate_optical_flow(frame)
        
        # Refine mask with optical flow (only after enough frames)
        if self.frame_count > self.window_size:
            refined_mask = self._refine_with_optical_flow(initial_mask, flow_magnitude)
        else:
            refined_mask = initial_mask
        
        # Apply closing to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small blobs
        refined_mask = self._remove_small_blobs(refined_mask, self.area_threshold)
        
        return refined_mask
    
    def _remove_small_blobs(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Remove small connected components below minimum area.
        
        Args:
            mask: Binary foreground mask
            min_area: Minimum blob area in pixels
            
        Returns:
            Cleaned mask
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        
        return out


def process_dynamic_background_sequence(
    frame_paths,
    out_dir: Path,
    max_frames: int = 300,
    window_size: int = 15,
    flow_threshold: float = 0.5,
    area_threshold: int = 300
):
    """
    Process a dynamic background sequence using hybrid algorithm.
    
    Args:
        frame_paths: List of frame paths
        out_dir: Output directory
        max_frames: Maximum frames to process (-1 for all)
        window_size: Temporal median window size
        flow_threshold: Optical flow magnitude threshold
        area_threshold: Minimum blob area
    """
    out_masks = out_dir / "masks"
    out_overlays = out_dir / "overlays"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    
    subtractor = HybridDynamicBackgroundSubtractor(
        window_size=window_size,
        flow_threshold=flow_threshold,
        area_threshold=area_threshold
    )
    
    n = len(frame_paths) if max_frames == -1 else min(len(frame_paths), max_frames)
    
    for i in tqdm(range(n), desc=str(out_dir)):
        frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        
        # Apply hybrid background subtraction
        mask = subtractor.apply(frame)
        
        # Create overlay visualization
        overlay = frame.copy()
        overlay[mask == 255] = np.clip(overlay[mask == 255] * 0.5 + 128, 0, 255).astype(np.uint8)
        
        # Save outputs
        cv2.imwrite(str(out_masks / f"{i:06d}.png"), mask)
        cv2.imwrite(str(out_overlays / f"{i:06d}.png"), overlay)
