from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from shadow_algorithm import process_shadow_sequence
from dynamic_background_algorithm import process_dynamic_background_sequence


def clean_mask(mog2_mask: np.ndarray, min_area: int = 300) -> np.ndarray:
    # Keep only sure foreground (remove shadows=127)
    mask = (mog2_mask == 255).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove small blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def process_sequence(frame_paths, out_dir: Path, max_frames: int = 300, category: str = None):
    """
    Process a sequence using the appropriate algorithm.
    
    For shadow sequences, uses specialized shadow-aware algorithm.
    For dynamic background sequences, uses hybrid temporal median + optical flow.
    For other sequences, uses standard MOG2 background subtraction.
    
    Args:
        frame_paths: List of frame paths
        out_dir: Output directory
        max_frames: Maximum frames to process
        category: Sequence category (optional, used to select algorithm)
    """
    # Check if this is a shadow sequence
    is_shadow = category == "shadow" or "shadow" in str(out_dir).lower()
    
    if is_shadow:
        # Use specialized shadow algorithm
        process_shadow_sequence(frame_paths, out_dir, max_frames=max_frames, save_shadow_maps=False)
        return
    
    # Check if this is a dynamic background sequence
    is_dynamic_bg = category == "dynamicBackground" or "dynamicBackground" in str(out_dir).lower()
    
    if is_dynamic_bg:
        # Use hybrid algorithm for dynamic backgrounds
        process_dynamic_background_sequence(frame_paths, out_dir, max_frames=max_frames)
        return
    
    # Standard processing for baseline, badWeather, PTZ sequences
    out_masks = out_dir / "masks"
    out_overlays = out_dir / "overlays"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    n = len(frame_paths) if max_frames == -1 else min(len(frame_paths), max_frames)

    for i in tqdm(range(n), desc=str(out_dir)):
        frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        raw = backsub.apply(frame)
        mask = clean_mask(raw, min_area=300)

        overlay = frame.copy()
        overlay[mask == 255] = np.clip(overlay[mask == 255] * 0.5 + 128, 0, 255)

        cv2.imwrite(str(out_masks / f"{i:06d}.png"), mask)
        cv2.imwrite(str(out_overlays / f"{i:06d}.png"), overlay.astype(np.uint8))
