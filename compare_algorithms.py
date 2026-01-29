"""
Compare optical flow only vs hybrid algorithm on dynamic backgrounds.
"""

import argparse
from pathlib import Path
from dataset_io import list_sequences, list_frames
from dynamic_background_algorithm import process_dynamic_background_sequence
from optical_flow_algorithm import process_optical_flow_sequence


def main():
    ap = argparse.ArgumentParser(description="Compare algorithms on dynamic background sequences")
    ap.add_argument("--category", default="dynamicBackground")
    ap.add_argument("--sequence", default="fountain01")
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--algorithm", choices=["hybrid", "optical_flow", "both"], default="both")
    args = ap.parse_args()

    frames = list_frames(args.category, args.sequence)

    if args.algorithm in ["optical_flow", "both"]:
        print(f"\n{'='*60}")
        print(f"OPTICAL FLOW ONLY - {args.category}/{args.sequence}")
        print(f"{'='*60}")
        out_dir = Path("outputs_comparison") / "optical_flow" / args.category / args.sequence
        process_optical_flow_sequence(frames, out_dir, max_frames=args.max_frames)
        print(f"Saved to: {out_dir.resolve()}\n")

    if args.algorithm in ["hybrid", "both"]:
        print(f"\n{'='*60}")
        print(f"HYBRID (Temporal Median + Optical Flow) - {args.category}/{args.sequence}")
        print(f"{'='*60}")
        out_dir = Path("outputs_comparison") / "hybrid" / args.category / args.sequence
        process_dynamic_background_sequence(frames, out_dir, max_frames=args.max_frames)
        print(f"Saved to: {out_dir.resolve()}\n")

    print("Comparison complete!")
    print("\nView results:")
    print(f"  Optical Flow: .\\\.venv\\Scripts\\python.exe view_outputs.py --category dynamicBackground --sequence {args.sequence} --max_frames {args.max_frames}")
    print(f"             (change outputs/ to outputs_comparison/optical_flow/)")
    print(f"  Hybrid:       .\\\.venv\\Scripts\\python.exe view_outputs.py --category dynamicBackground --sequence {args.sequence} --max_frames {args.max_frames}")
    print(f"             (change outputs/ to outputs_comparison/hybrid/)")


if __name__ == "__main__":
    main()
