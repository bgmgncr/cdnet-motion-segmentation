"""
Pure optical flow-based motion segmentation for dynamic backgrounds.

Uses only optical flow (Farneback) to detect moving objects.
Simpler than temporal median + optical flow hybrid.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class OpticalFlowBackgroundSubtractor:
    """
    Background subtraction using pure optical flow.
    
    Detects moving objects by analyzing motion magnitude and consistency
    between consecutive frames.
    """
    
    def __init__(self, flow_threshold=1.0, area_threshold=300):
        """
        Args:
            flow_threshold: Optical flow magnitude threshold for motion detection
            area_threshold: Minimum blob size to keep
        """
        self.flow_threshold = flow_threshold
        self.area_threshold = area_threshold
        
        # Previous frame for optical flow calculation
        self.prev_frame_gray = None
        self.frame_count = 0
    
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
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply optical flow-based background subtraction.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask (0=background, 255=foreground)
        """
        self.frame_count += 1
        
        # Get optical flow magnitude
        flow_magnitude = self._calculate_optical_flow(frame)
        
        # In early frames, flow is unreliable - be more permissive
        if self.frame_count < 10:
            threshold = self.flow_threshold * 0.5
        else:
            threshold = self.flow_threshold
        
        # Threshold: pixels with significant motion = foreground
        mask = (flow_magnitude > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply closing to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small blobs
        mask = self._remove_small_blobs(mask, self.area_threshold)
        
        return mask
    
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


def process_optical_flow_sequence(
    frame_paths,
    out_dir: Path,
    max_frames: int = 300,
    flow_threshold: float = 1.0,
    area_threshold: int = 300
):
    """
    Process a dynamic background sequence using pure optical flow.
    
    Args:
        frame_paths: List of frame paths
        out_dir: Output directory
        max_frames: Maximum frames to process (-1 for all)
        flow_threshold: Optical flow magnitude threshold
        area_threshold: Minimum blob area
    """
    out_masks = out_dir / "masks"
    out_overlays = out_dir / "overlays"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    
    subtractor = OpticalFlowBackgroundSubtractor(
        flow_threshold=flow_threshold,
        area_threshold=area_threshold
    )
    
    n = len(frame_paths) if max_frames == -1 else min(len(frame_paths), max_frames)
    
    for i in tqdm(range(n), desc=str(out_dir)):
        frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        
        # Apply optical flow background subtraction
        mask = subtractor.apply(frame)
        
        # Create overlay visualization
        overlay = frame.copy()
        overlay[mask == 255] = np.clip(overlay[mask == 255] * 0.5 + 128, 0, 255).astype(np.uint8)
        
        # Save outputs
        cv2.imwrite(str(out_masks / f"{i:06d}.png"), mask)
        cv2.imwrite(str(out_overlays / f"{i:06d}.png"), overlay)
