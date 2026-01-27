from pathlib import Path
from dataset_io import list_categories, list_sequences, list_frames
from pipeline import process_sequence


def main():
    category = "baseline"

    cats = list_categories()
    if category not in cats:
        raise ValueError(f"Category '{category}' not found. Available: {cats}")

    seqs = list_sequences(category)
    if not seqs:
        raise RuntimeError(f"No sequences found in dataset/{category}")

    sequence = seqs[0]  # first baseline sequence
    frames = list_frames(category, sequence)

    out_dir = Path("outputs") / category / sequence
    print(f"Running on: {category}/{sequence}")
    print(f"Frames found: {len(frames)}")
    process_sequence(frames, out_dir, max_frames=300)  # first test: 300 frames
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
