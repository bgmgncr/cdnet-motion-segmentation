"""
Test and visualize shadow detection algorithm.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from dataset_io import list_sequences, list_frames
from shadow_algorithm import ShadowAwareBackgroundSubtractor


def main():
    ap = argparse.ArgumentParser(description="Visualize shadow detection")
    ap.add_argument("--sequence", default="peopleInShade", help="Shadow sequence to visualize")
    ap.add_argument("--max_frames", type=int, default=100, help="Maximum frames to process")
    ap.add_argument("--show_shadow_map", action="store_true", help="Show shadow probability map")
    args = ap.parse_args()

    # Get shadow sequence frames
    sequences = list_sequences("shadow")
    if args.sequence not in sequences:
        print(f"Available shadow sequences: {sequences}")
        raise ValueError(f"Sequence '{args.sequence}' not found")

    frames = list_frames("shadow", args.sequence)
    n = min(len(frames), args.max_frames)

    subsub = ShadowAwareBackgroundSubtractor()

    print(f"Processing: shadow/{args.sequence} ({n} frames)")
    print("Controls: SPACE=pause/play, ESC=quit, 'S'=toggle shadow map")

    paused = False
    show_shadow = args.show_shadow_map

    for i in range(n):
        frame = cv2.imread(str(frames[i]), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Get detections
        mask = subsub.apply(frame)
        shadow_map = subsub.get_shadow_map(frame) if show_shadow else None

        # Create visualization
        overlay = frame.copy()
        if mask is not None:
            overlay[mask == 255] = np.clip(overlay[mask == 255] * 0.5 + 128, 0, 255).astype(np.uint8)

        # Combine displays
        if show_shadow and shadow_map is not None:
            shadow_map_3ch = cv2.cvtColor(shadow_map, cv2.COLOR_GRAY2BGR)
            display = np.hstack([overlay, shadow_map_3ch])
            text = f"Frame {i}/{n} | Shadow Map ON | SPACE=pause, S=toggle, ESC=quit"
        else:
            display = overlay
            text = f"Frame {i}/{n} | Shadow Map OFF | SPACE=pause, S=toggle, ESC=quit"

        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Shadow Detection Visualization", display)

        while True:
            key = cv2.waitKey(30 if not paused else 0)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):  # SPACE
                paused = not paused
                break
            elif key == ord('s') or key == ord('S'):
                show_shadow = not show_shadow
                break
            elif key == -1:  # No key pressed
                break

    cv2.destroyAllWindows()
    print("Visualization complete!")


if __name__ == "__main__":
    main()
