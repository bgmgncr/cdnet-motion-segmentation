import argparse
from pathlib import Path

from dataset_io import list_categories, list_sequences, list_frames
from pipeline import process_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="baseline")
    ap.add_argument("--sequence", default=None, help="If omitted, uses first sequence in the category")
    ap.add_argument("--max_frames", type=int, default=300, help="Use -1 for all frames")

    ap.add_argument("--list", action="store_true", help="List categories and exit")
    ap.add_argument("--list_sequences", action="store_true", help="List sequences in category and exit")

    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    if args.list:
        print("Categories:", list_categories())
        return

    cats = list_categories()
    if args.category not in cats:
        raise ValueError(f"Category '{args.category}' not found. Available: {cats}")

    if args.list_sequences:
        print(f"Sequences in {args.category}:", list_sequences(args.category))
        return

    seqs = list_sequences(args.category)
    if not seqs:
        raise RuntimeError(f"No sequences found in {args.category}")

    seq = args.sequence or seqs[0]
    frames = list_frames(args.category, seq)

    out_dir = Path(args.out_dir) / args.category / seq
    print(f"Running on: {args.category}/{seq}")
    print(f"Frames found: {len(frames)}")

    process_sequence(frames, out_dir, max_frames=args.max_frames)

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
