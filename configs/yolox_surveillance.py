"""
yolox_surveillance.py — YOLOX-s Experiment Config for Surveillance Detection

Model choice: YOLOX-s (Small)
  - Depth=0.33, Width=0.50 → ~9M parameters
  - Input: 640×640
  - COCO mAP: 40.5  Speed: ~25ms/frame on GPU
  - Fits comfortably in 6 GB VRAM with batch=8 + FP16
  - Best trade-off between speed and accuracy for real-time CCTV inference

Classes: 0=fire, 1=weapon
Dataset: COCO-format JSON annotations from merged_dataset/
"""

import os
from pathlib import Path
from yolox.exp import Exp as BaseExp

# --------------------------------------------------------------------------- #
# Dataset paths
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_ROOT = _REPO_ROOT / "merged_dataset"


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------- #
        # Model architecture — YOLOX-s
        # ------------------------------------------------------------------- #
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        # ------------------------------------------------------------------- #
        # Dataset
        # ------------------------------------------------------------------- #
        self.num_classes = 2
        self.data_dir = str(_DATASET_ROOT)
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        # ------------------------------------------------------------------- #
        # Training hyperparameters
        # ------------------------------------------------------------------- #
        self.max_epoch = 100
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (10, 20)

        # Batch size: 8 fits in 6 GB VRAM with FP16
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0  # scaled per image
        self.scheduler = "yoloxwarmcos"
        self.warmup_lr = 0.0
        self.min_lr_ratio = 0.05

        # ------------------------------------------------------------------- #
        # Augmentation
        # ------------------------------------------------------------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2.0)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True
        self.no_aug_epochs = 15

        # ------------------------------------------------------------------- #
        # Training efficiency
        # ------------------------------------------------------------------- #
        self.data_num_workers = 2   # Windows: keep low to avoid multiprocessing issues
        self.ema = True
        self.print_interval = 10    # Print every 10 iterations for live log streaming
        self.eval_interval = 5      # Evaluate every 5 epochs
        self.save_history_ckpt = False  # Only save best + last to save disk

        # ------------------------------------------------------------------- #
        # Output
        # ------------------------------------------------------------------- #
        self.output_dir = str(_REPO_ROOT / "models")
        self.experiment_name = "yolox_surveillance"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """
        Override to point images at merged_dataset/train/images/<file_name>
        YOLOX COCODataset loads: data_dir / name / file_name
        So name='train/images' gives: merged_dataset/train/images/<file_name>
        """
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="train/images",   # resolved: merged_dataset/train/images/<file>
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        train_loader = DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=worker_init_reset_seed,
        )
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Override to point images at merged_dataset/val/images/<file_name>
        """
        import torch
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val/images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        sampler = torch.utils.data.SequentialSampler(valdataset)

        val_loader = torch.utils.data.DataLoader(
            valdataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
        )
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
