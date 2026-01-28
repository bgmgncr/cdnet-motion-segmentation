"""
Shadow-aware motion segmentation algorithm.

Uses multiple color spaces (BGR, HSV, YUV) and specialized shadow detection
to accurately identify moving objects while reducing shadow artifacts.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class ShadowAwareBackgroundSubtractor:
    """
    Specialized background subtraction algorithm for shadow handling.
    
    Combines MOG2 with HSV and YUV analysis to distinguish between
    moving objects and shadow variations.
    """
    
    def __init__(self, history=300, var_threshold=20):
        self.history = history
        self.var_threshold = var_threshold
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold, 
            detectShadows=True
        )
        self.frame_count = 0
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply shadow-aware background subtraction.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask with 0=background, 255=foreground, 127=shadow (removed)
        """
        self.frame_count += 1
        
        # Get MOG2 output (255=foreground, 127=shadow, 0=background)
        mog2_mask = self.mog2.apply(frame)
        
        # In early frames, MOG2 is still learning - be more permissive
        if self.frame_count < 50:
            # Include both confirmed foreground and shadows
            foreground = ((mog2_mask == 255) | (mog2_mask == 127)).astype(np.uint8) * 255
            return foreground
        
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Convert frame to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
        
        # Extract channels
        h, s, v = cv2.split(hsv)
        y, u, uv = cv2.split(yuv)
        
        # Shadow detection using multiple color space properties
        # Shadows typically have:
        # - Similar hue (color) but reduced saturation and value
        # - Reduced Y (brightness) in YUV
        # - Normal or similar chrominance (U, V)
        
        # 1. HSV-based shadow detection
        # Shadows have low saturation changes but low value
        hsv_shadow_score = np.zeros_like(h, dtype=np.float32)
        
        # Low saturation suggests potential shadow
        hsv_shadow_score += (s < 30) * 0.3
        
        # Very low value (dark) but not completely black
        hsv_shadow_score += ((v > 20) & (v < 100)) * 0.3
        
        # 2. YUV-based shadow detection
        yuv_shadow_score = np.zeros_like(y, dtype=np.float32)
        
        # Reduced brightness characteristic of shadows
        yuv_shadow_score += (y < 100) * 0.4
        
        # Low chrominance variation (U, V close to 128)
        u_dev = np.abs(u - 128)
        v_dev = np.abs(v - 128)
        yuv_shadow_score += ((u_dev < 20) & (v_dev < 20)) * 0.3
        
        # Combined shadow score
        shadow_score = (hsv_shadow_score + yuv_shadow_score) / 2
        
        # Process MOG2 mask
        # Keep only sure foreground (255) and remove shadows (127)
        foreground = (mog2_mask == 255).astype(np.uint8)
        
        # Additional filtering: if shadow score is high, reduce confidence
        shadow_mask = (shadow_score > 0.5).astype(np.uint8)
        
        # Adjust foreground based on shadow detection
        # True foreground should have lower shadow score
        foreground[shadow_mask == 1] = 0
        
        return foreground * 255
    
    def get_shadow_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Get shadow map for visualization/debugging.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Shadow probability map (0-255)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
        
        h, s, v = cv2.split(hsv)
        y, u, uv = cv2.split(yuv)
        
        hsv_shadow_score = np.zeros_like(h, dtype=np.float32)
        hsv_shadow_score += (s < 30) * 0.3
        hsv_shadow_score += ((v > 20) & (v < 100)) * 0.3
        
        yuv_shadow_score = np.zeros_like(y, dtype=np.float32)
        yuv_shadow_score += (y < 100) * 0.4
        u_dev = np.abs(u - 128)
        v_dev = np.abs(v - 128)
        yuv_shadow_score += ((u_dev < 20) & (v_dev < 20)) * 0.3
        
        shadow_score = (hsv_shadow_score + yuv_shadow_score) / 2
        return (shadow_score * 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """
    Clean foreground mask using enhanced morphological operations.
    
    Args:
        mask: Binary foreground mask
        min_area: Minimum area to keep (removes noise)
        
    Returns:
        Cleaned mask
    """
    # Stronger opening to remove thin noise
    # Use larger kernel and more iterations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Aggressive opening to remove noise and thin structures
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large, iterations=1)
    
    # Closing to fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # Apply median blur to the binary mask for smoothing
    mask = cv2.medianBlur(mask, 5)
    
    # Remove small blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    
    # Final median blur for smoothing edges
    out = cv2.medianBlur(out, 3)
    
    return out


def process_shadow_sequence(
    frame_paths, 
    out_dir: Path, 
    max_frames: int = 300,
    min_area: int = 500,
    save_shadow_maps: bool = False
):
    """
    Process a shadow sequence using the shadow-aware algorithm.
    
    Args:
        frame_paths: List of input frame paths
        out_dir: Output directory for masks and overlays
        max_frames: Maximum frames to process (-1 for all)
        min_area: Minimum connected component area
        save_shadow_maps: Whether to save shadow maps for debugging
    """
    out_masks = out_dir / "masks"
    out_overlays = out_dir / "overlays"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    
    if save_shadow_maps:
        out_shadow_maps = out_dir / "shadow_maps"
        out_shadow_maps.mkdir(parents=True, exist_ok=True)
    
    subsub = ShadowAwareBackgroundSubtractor(history=300, var_threshold=20)
    
    n = len(frame_paths) if max_frames == -1 else min(len(frame_paths), max_frames)
    
    for i in tqdm(range(n), desc=f"Processing shadow sequence: {out_dir.name}"):
        frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        
        # Get foreground mask
        raw_mask = subsub.apply(frame)
        mask = clean_mask(raw_mask, min_area=min_area)
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask == 255] = np.clip(overlay[mask == 255] * 0.5 + 128, 0, 255)
        
        # Save outputs
        cv2.imwrite(str(out_masks / f"{i:06d}.png"), mask)
        cv2.imwrite(str(out_overlays / f"{i:06d}.png"), overlay.astype(np.uint8))
        
        # Optionally save shadow map
        if save_shadow_maps:
            shadow_map = subsub.get_shadow_map(frame)
            cv2.imwrite(str(out_shadow_maps / f"{i:06d}.png"), shadow_map)
