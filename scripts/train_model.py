"""
train_model.py — YOLOX-s Real-Time Surveillance Training Pipeline

Migrated from RTMDet/MMDetection to pure PyTorch YOLOX-s.
Zero compilation required. Runs natively on Windows with PyTorch 2.4 + CUDA.

Features:
  - Automatically verifies/converts YOLO format annotations to COCO JSON format.
  - Automatically downloads pre-trained YOLOX-s COCO weights for transfer learning.
  - Enforces FP16 Automatic Mixed Precision (AMP) for NVIDIA RTX 3050 (6 GB VRAM).
  - High-frequency live console logging (every 10 batches) + TensorBoard.
"""

import os
import sys
import urllib.request
from pathlib import Path

import torch
from yolox.core import Trainer
from yolox.tools.train import make_parser

# Add repository root to path so we can import configs and scripts
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.yolox_surveillance import Exp
from scripts.convert_yolo_to_coco import convert_dataset_root, DEFAULT_CLASSES


def ensure_coco_annotations(dataset_root: Path) -> None:
    """Verifies COCO JSON annotations exist; if not, converts YOLO label files."""
    print("\n[*] Step 1: Verifying dataset annotations...")
    annotations_dir = dataset_root / "annotations"
    train_json = annotations_dir / "train.json"
    val_json = annotations_dir / "val.json"

    if not train_json.exists() or not val_json.exists():
        print("[!] COCO JSON annotations missing. Automatically converting from YOLO format...")
        convert_dataset_root(dataset_root, DEFAULT_CLASSES)
        print("[+] Dataset conversion complete!")
    else:
        print(f"[+] Found existing COCO annotations: {train_json.as_posix()}")


def ensure_pretrained_backbone(models_dir: Path) -> Path:
    """Downloads official YOLOX-s pre-trained COCO backbone weights if not present."""
    print("\n[*] Step 2: Verifying pre-trained YOLOX-s backbone weights...")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "yolox_s.pth"

    if not ckpt_path.exists():
        url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
        print(f"[*] Downloading YOLOX-s pre-trained weights from Megvii CDN:\n    {url}")
        urllib.request.urlretrieve(url, ckpt_path)
        print(f"[+] Download complete: {ckpt_path.as_posix()} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[+] Found pre-trained weights: {ckpt_path.as_posix()}")

    return ckpt_path


def main():
    print("=" * 60)
    print("YOLOX-s Real-Time Surveillance Training Pipeline")
    print("=" * 60)

    # 1. Check hardware & FP16 support
    if not torch.cuda.is_available():
        print("[!] WARNING: CUDA is not available! Training will fall back to CPU and be extremely slow.")
        device_name = "CPU"
    else:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] Hardware Detected: {device_name} ({vram_gb:.1f} GB VRAM)")
        print("[*] Enforcing CUDA device execution (cuda:0) with FP16 Mixed Precision.")

    dataset_root = _REPO_ROOT / "merged_dataset"
    models_dir = _REPO_ROOT / "models"

    # 2. Prepare data & checkpoint
    ensure_coco_annotations(dataset_root)
    ckpt_path = ensure_pretrained_backbone(models_dir)

    # 3. Initialize YOLOX Experiment & Training Arguments
    print("\n[*] Step 3: Initializing YOLOX-s training engine...")
    exp = Exp()
    args = make_parser().parse_args([])
    args.experiment_name = exp.experiment_name
    args.name = "yolox-s"
    args.batch_size = 8       # Batch size 8 fits comfortably in 6 GB VRAM in FP16
    args.devices = 1
    args.fp16 = True          # Enable Automatic Mixed Precision (AMP)
    args.ckpt = str(ckpt_path)
    args.logger = "tensorboard"
    args.occupy = False
    args.cache = False        # Disabled on Windows (no fork support)

    print(f"[*] Experiment Name: {exp.experiment_name}")
    print(f"[*] Target Epochs:   {exp.max_epoch}")
    print(f"[*] Batch Size:      {args.batch_size}")
    print(f"[*] Live Log Freq:   Every {exp.print_interval} batches")
    print("=" * 60)

    # 4. Launch Training
    trainer = Trainer(exp, args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Best/latest checkpoint preserved in models/yolox_surveillance/")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[!] Training failed with error: {exc}")
        raise


if __name__ == "__main__":
    main()