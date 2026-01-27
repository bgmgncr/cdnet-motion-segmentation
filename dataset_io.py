from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset"

IMAGE_EXTS = (".jpg", ".png", ".bmp")


def list_categories():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"dataset folder not found: {DATA_ROOT}")
    return sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])


def list_sequences(category: str):
    cat_dir = DATA_ROOT / category
    if not cat_dir.exists():
        raise FileNotFoundError(f"Category not found: {cat_dir}")
    return sorted([p.name for p in cat_dir.iterdir() if p.is_dir()])


def list_frames(category: str, sequence: str):
    inp = DATA_ROOT / category / sequence / "input"
    if not inp.exists():
        raise FileNotFoundError(f"Missing input folder: {inp}")

    frames = sorted([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not frames:
        raise FileNotFoundError(f"No frames found in: {inp}")
    return frames
