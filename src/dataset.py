"""Optimized PyTorch Dataset for the Face-to-BMI project (W3, Role 2).

Consumes:
  - data/processed/clean_data.csv    schema: [Unnamed: 0, bmi, gender, is_training, name]
  - data/processed/faces_standardized/img_<image_id>.bmp  (224x224 RGB)

Defaults auto-detect the team Google Drive mount when running on Colab:
  /content/drive/MyDrive/ML2-Team2/Final Project/face-to-bmi/data
otherwise fall back to the repo-relative `data/` directory.

Provides:
  - FaceBMIDataset(split, image_dir, cache, transform)
  - eval_transform() / train_transform()
  - build_dataloader(dataset, ...)
  - python -m src.dataset --benchmark
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ML2-Team2/Final Project/face-to-bmi/data")


def _data_root() -> Path:
    return DRIVE_DATA_ROOT if DRIVE_DATA_ROOT.exists() else REPO_ROOT / "data"


DEFAULT_DATA_ROOT = _data_root()
DEFAULT_LABEL_CSV = DEFAULT_DATA_ROOT / "processed" / "clean_data.csv"
DEFAULT_IMAGE_DIR = DEFAULT_DATA_ROOT / "processed" / "faces_standardized"
DEFAULT_MMAP_PATH = DEFAULT_DATA_ROOT / "processed" / "faces_standardized.uint8.npy"

_IMG_ID_RE = re.compile(r"img_(\d+)\.bmp$", re.IGNORECASE)

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CacheMode = Literal["none", "ram", "mmap"]
SplitName = Literal["train", "test", "all"]


def eval_transform() -> Callable:
    """Matches Ryan's feature-extraction-v1.ipynb pipeline byte-for-byte."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def train_transform() -> Callable:
    """Eval transform + mild augmentation suitable for pre-aligned 224x224 face crops."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FaceBMIDataset(Dataset):
    def __init__(
        self,
        split: SplitName = "train",
        image_dir: Path | str = DEFAULT_IMAGE_DIR,
        label_csv: Path | str = DEFAULT_LABEL_CSV,
        cache: CacheMode = "ram",
        transform: Optional[Callable] = None,
        mmap_path: Path | str = DEFAULT_MMAP_PATH,
    ):
        self.image_dir = Path(image_dir)
        self.cache = cache
        self.transform = transform if transform is not None else eval_transform()

        df = pd.read_csv(label_csv)
        if split == "train":
            df = df[df["is_training"] == 1]
        elif split == "test":
            df = df[df["is_training"] == 0]
        df = df.reset_index(drop=True)
        self.metadata = df

        self._names = df["name"].to_numpy()
        self._bmis = df["bmi"].to_numpy(dtype=np.float32)
        self._genders = df["gender"].to_numpy()
        self._image_ids = np.array(
            [int(m.group(1)) if (m := _IMG_ID_RE.search(str(n))) else -1 for n in self._names],
            dtype=np.int64,
        )

        self._ram_buf: Optional[np.ndarray] = None
        self._mmap_buf: Optional[np.ndarray] = None

        if cache == "ram":
            self._ram_buf = self._decode_all_to_uint8()
        elif cache == "mmap":
            self._mmap_buf = self._load_or_build_mmap(Path(mmap_path))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        if self.cache == "ram":
            arr = self._ram_buf[idx]                # (H, W, 3) uint8
            image = Image.fromarray(arr, mode="RGB")
        elif self.cache == "mmap":
            arr = np.array(self._mmap_buf[idx])     # copy out of mmap before PIL
            image = Image.fromarray(arr, mode="RGB")
        else:
            image = Image.open(self.image_dir / self._names[idx]).convert("RGB")

        image = self.transform(image)

        return {
            "image": image,
            "bmi": torch.tensor(float(self._bmis[idx]), dtype=torch.float32),
            "image_id": int(self._image_ids[idx]),
            "name": str(self._names[idx]),
            "gender": str(self._genders[idx]),
        }

    def _decode_all_to_uint8(self) -> np.ndarray:
        n = len(self._names)
        buf = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i, name in enumerate(self._names):
            with Image.open(self.image_dir / name) as im:
                im = im.convert("RGB")
                if im.size != (IMG_SIZE, IMG_SIZE):
                    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                buf[i] = np.asarray(im, dtype=np.uint8)
        return buf

    def _load_or_build_mmap(self, mmap_path: Path) -> np.ndarray:
        if not mmap_path.exists():
            mmap_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mmap_path, self._decode_all_to_uint8())
        arr = np.load(mmap_path, mmap_mode="r")
        if len(arr) != len(self._names):
            raise RuntimeError(
                f"mmap cache at {mmap_path} has {len(arr)} rows but split has "
                f"{len(self._names)}; delete the .npy and rebuild."
            )
        return arr


def build_dataloader(
    dataset: FaceBMIDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
    )


def benchmark_throughput(
    split: SplitName = "train",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Compare images/sec across cache modes for one full epoch.

    Run after Drive sync so data/processed/faces_standardized/ is populated.
    """
    results = {}
    for cache in ("none", "ram"):
        ds = FaceBMIDataset(split=split, cache=cache)
        loader = build_dataloader(ds, batch_size=batch_size, num_workers=num_workers)
        n = len(ds)
        start = time.perf_counter()
        for batch in loader:
            _ = batch["image"].shape
        elapsed = time.perf_counter() - start
        ips = n / elapsed
        results[cache] = {"images": n, "seconds": round(elapsed, 3), "images_per_sec": round(ips, 1)}
        print(f"cache={cache:5s}  {n} imgs in {elapsed:6.2f}s  ->  {ips:7.1f} img/s")
    if "ram" in results and "none" in results:
        speedup = results["ram"]["images_per_sec"] / max(results["none"]["images_per_sec"], 1e-6)
        print(f"speedup (ram / none): {speedup:.2f}x")
        results["speedup"] = round(speedup, 2)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.benchmark:
        benchmark_throughput(split=args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        ds = FaceBMIDataset(split=args.split, cache="none")
        sample = ds[0]
        print(f"split={args.split}  len={len(ds)}  image={sample['image'].shape}  bmi={sample['bmi'].item():.2f}")


if __name__ == "__main__":
    _main()
