from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def resolve_split_root(dataset_root: Path, split: str) -> Path:
    if (dataset_root / split).is_dir():
        return dataset_root / split
    direct_subdirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()] if dataset_root.exists() else []
    if len(direct_subdirs) == 1 and (direct_subdirs[0] / split).is_dir():
        return direct_subdirs[0] / split
    return dataset_root / split


def sample_split(source_root: Path, target_root: Path, per_class: int, seed: int) -> None:
    if not source_root.exists():
        return

    rng = random.Random(seed)
    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue

        files = [path for path in sorted(class_dir.iterdir()) if path.is_file()]
        if not files:
            continue

        count = min(per_class, len(files))
        sampled = rng.sample(files, count)
        destination = target_root / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        for file_path in sampled:
            shutil.copy2(file_path, destination / file_path.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--val-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)

    train_source = resolve_split_root(source_root, "train")
    val_source = resolve_split_root(source_root, "val")

    if output_root.exists():
        shutil.rmtree(output_root)

    sample_split(train_source, output_root / "train", args.train_per_class, args.seed)
    sample_split(val_source, output_root / "val", args.val_per_class, args.seed + 1)

    print(f"Reduced dataset created in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
