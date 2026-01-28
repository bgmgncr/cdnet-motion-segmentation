"""
Flexible sequence viewer for outputs from any category and sequence.
"""

import argparse
import cv2
import glob
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="View motion segmentation outputs")
    ap.add_argument("--category", default="shadow", help="Category (e.g., shadow, baseline, dynamicBackground)")
    ap.add_argument("--sequence", default="backdoor", help="Sequence name")
    ap.add_argument("--max_frames", type=int, default=-1, help="Max frames to view (-1 for all)")
    ap.add_argument("--output_type", choices=["overlays", "masks"], default="overlays", help="View overlays or masks")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second for playback")
    args = ap.parse_args()

    output_dir = Path("outputs") / args.category / args.sequence / args.output_type
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        print(f"\nAvailable categories: baseline, shadow, badWeather, dynamicBackground, PTZ")
        
        # List available sequences in the selected category
        cat_dir = Path("outputs") / args.category
        if cat_dir.exists():
            available_seqs = [d.name for d in cat_dir.iterdir() if d.is_dir()]
            print(f"\nAvailable sequences in '{args.category}':")
            for seq in sorted(available_seqs):
                seq_dir = cat_dir / seq
                has_overlays = (seq_dir / "overlays").exists() and len(list((seq_dir / "overlays").glob("*.png"))) > 0
                has_masks = (seq_dir / "masks").exists() and len(list((seq_dir / "masks").glob("*.png"))) > 0
                status = "✓" if (has_overlays or has_masks) else "✗"
                print(f"  {status} {seq} (overlays: {has_overlays}, masks: {has_masks})")
        return

    # Get all frame files
    paths = sorted(glob.glob(str(output_dir / "*.png")))
    
    if not paths:
        print(f"No frames found in {output_dir}")
        return

    if args.max_frames > 0:
        paths = paths[:args.max_frames]

    print(f"Viewing: {args.category}/{args.sequence}")
    print(f"Output type: {args.output_type}")
    print(f"Total frames: {len(paths)}")
    print("Controls: SPACE=pause/play, LEFT/RIGHT=frame navigation, ESC=quit")
    print()

    current_frame = 0
    paused = False
    delay_ms = max(1, 1000 // args.fps)

    while current_frame < len(paths):
        img = cv2.imread(paths[current_frame])
        if img is None:
            current_frame += 1
            continue

        # Create display with info
        display = img.copy()
        text = f"Frame {current_frame + 1}/{len(paths)} | {args.output_type} | SPACE=pause, ARROWS=nav, ESC=quit"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f"{args.category}/{args.sequence}", display)

        while True:
            key = cv2.waitKey(delay_ms if not paused else 0)
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                print("Viewer closed.")
                return
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"{'Paused' if paused else 'Playing'} at frame {current_frame + 1}")
                break
            elif key == 81 or key == 2:  # LEFT arrow or A key
                current_frame = max(0, current_frame - 1)
                print(f"Jumped to frame {current_frame + 1}")
                break
            elif key == 83 or key == 3:  # RIGHT arrow or D key
                current_frame = min(len(paths) - 1, current_frame + 1)
                print(f"Jumped to frame {current_frame + 1}")
                break
            elif key == -1:  # No key pressed
                if not paused:
                    break
                # If paused, keep waiting
            else:
                break

        if not paused:
            current_frame += 1

    cv2.destroyAllWindows()
    print("Playback complete!")


if __name__ == "__main__":
    main()
