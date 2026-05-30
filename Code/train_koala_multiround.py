#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_koala_multiround.py

Multi-round Faster R-CNN training pipeline for the Koala Be Where project.

The main idea is to train a koala detector with an additional Sobel edge channel
and then repeatedly mine hard-negative examples.

Pipeline overview:
1. Train an initial koala detector using positive koala annotations.
2. Run the current model over negative / background images.
3. Collect false-positive detections as hard negatives.
4. Build a balanced dataset of koala positives and mined negatives.
5. Train the next round of the detector.
6. Repeat until no useful hard negatives are found.

Model design:
- Backbone: Faster R-CNN with ResNet-50 FPN.
- Input channels: 4 channels instead of 3.
  - RGB channels provide colour and texture.
  - The fourth channel is a Sobel edge magnitude map.
- Classes:
  - 0 = background, handled internally by Faster R-CNN
  - 1 = koala
  - 2 = negative / confusing background

This project was designed for practical wildlife detection where koalas may be
partially hidden by trees, branches, foliage, lighting changes, and cluttered
natural backgrounds.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
from collections import deque
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Iterable, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform


# =============================================================================
# Default configuration
# =============================================================================

DEFAULT_IMAGES_DIR = "/Users/richman/Downloads/archive/Koalas"
DEFAULT_ALT_IMAGES_DIR = "/Users/richman/Downloads/openimages"
DEFAULT_ORIG_ANNOTATIONS = "/Users/richman/Downloads/annotations_updated.csv"
DEFAULT_OUTPUT_DIR = "."

DEFAULT_DEVICE = "cpu"

POS_ANNOTATIONS = "annotations_pos.csv"
ANNOTATIONS_ROUND_TEMPLATE = "annotations_round{round}.csv"
MODEL_ROUND_TEMPLATE = "model_round{round}.pth"
NEG_FOLDER_TEMPLATE = "hard_neg_round{round}"
NEG_CSV_TEMPLATE = "hard_neg_round{round}.csv"

BATCH_SIZE = 15
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-2
STEP_SIZE = 3
GAMMA = 0.1

# Hull Moving Average based early-stopping settings.
HMA_WINDOW = 5
HMA_SLOPE_THRESHOLD = 1e-2
HMA_STOP_PATIENCE = 3

DETECTION_THRESHOLD = 0.5
NUM_WORKERS = cpu_count()

IMAGE_MEAN4 = (0.485, 0.456, 0.406, 0.0)
IMAGE_STD4 = (0.229, 0.224, 0.225, 1.0)


# Globals used by multiprocessing hard-negative workers.
_WORKER_MODEL = None
_WORKER_ORIG_DF = None
_WORKER_CONFIG = {}


# =============================================================================
# Moving-average utilities for early stopping
# =============================================================================

def weighted_moving_average(values: Iterable[float], window: int) -> np.ndarray:
    """Compute a weighted moving average with linearly increasing weights."""
    values = np.asarray(values, dtype=float)
    weights = np.arange(1, window + 1, dtype=float)
    return np.convolve(values, weights / weights.sum(), mode="valid")


def hull_moving_average(values: Iterable[float], window: int) -> np.ndarray:
    """Compute the Hull Moving Average used to smooth validation loss.

    The Hull Moving Average is calculated as:

        HMA(n) = WMA(2 * WMA(values, n/2) - WMA(values, n), sqrt(n))

    In this training loop it is used to reduce noisy validation-loss decisions.
    Early stopping is triggered when the smoothed loss stops improving for
    several consecutive checks.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)

    half_window = window // 2
    wma_half = weighted_moving_average(values, half_window)
    wma_full = weighted_moving_average(values, window)

    difference = 2 * wma_half[-len(wma_full):] - wma_full
    hma = weighted_moving_average(difference, int(math.sqrt(window)))

    padding = len(values) - len(hma)
    return np.concatenate([np.full(padding, np.nan), hma])


# =============================================================================
# Image preprocessing and dataset
# =============================================================================

def compute_sobel_channel(image_rgb: np.ndarray) -> np.ndarray:
    """Compute a single Sobel edge-magnitude channel from an RGB image.

    The Sobel operator is applied to each RGB channel separately. The maximum
    edge magnitude across the three channels is used as the fourth channel.

    This gives the detector an explicit edge/shape signal, which can be useful
    when the animal blends into natural colours or is partially occluded.
    """
    sobel_channels = []

    for channel_index in range(3):
        channel = image_rgb[:, :, channel_index]
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_channels.append(magnitude)

    sobel_max = np.max(np.stack(sobel_channels, axis=0), axis=0)
    sobel_uint8 = (sobel_max / (sobel_max.max() + 1e-6) * 255.0).astype(np.uint8)
    return sobel_uint8[:, :, None]


def build_four_channel_image(image_path: str) -> np.ndarray:
    """Load an RGB image and append the Sobel edge channel."""
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    sobel_channel = compute_sobel_channel(image_rgb)
    return np.concatenate([image_rgb, sobel_channel], axis=2)


def get_transform(train: bool) -> A.Compose:
    """Create Albumentations transforms for training or validation."""
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=0),
                A.Normalize(mean=(0, 0, 0, 0), std=(1, 1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    return A.Compose(
        [
            A.Normalize(mean=(0, 0, 0, 0), std=(1, 1, 1, 1), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class KoalaDataset(Dataset):
    """Dataset for koala positive images and mined hard-negative images.

    The annotation CSV must contain:

        image_name,class,xmin,ymin,xmax,ymax,normalized

    The `normalized` column controls image location and box scaling:
    - "yes": image is loaded from the Open Images folder and boxes are normalised.
    - "no": image is loaded from the original koala archive and boxes are absolute.
    - any other value: treated as a folder name, usually `hard_neg_roundX`.
    """

    REQUIRED_COLUMNS = ["image_name", "class", "xmin", "ymin", "xmax", "ymax", "normalized"]

    def __init__(
        self,
        images_dir: str,
        alt_images_dir: str,
        csv_path: str,
        transforms: A.Compose | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.alt_images_dir = alt_images_dir
        self.transforms = transforms

        self.df = pd.read_csv(csv_path)

        for column in self.REQUIRED_COLUMNS:
            if column not in self.df.columns:
                raise ValueError(f"Annotation CSV missing required column: {column}")

        self.image_names = sorted(self.df["image_name"].unique())

    def __len__(self) -> int:
        return len(self.image_names)

    def resolve_image_root(self, normalized_value: str) -> str:
        """Resolve the correct image folder based on the annotation source flag."""
        value = str(normalized_value).strip().lower()

        if value == "yes":
            return self.alt_images_dir

        if value == "no":
            return self.images_dir

        # For mined negatives, the value is the folder name itself.
        return value

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_name = self.image_names[idx]
        records = self.df[self.df["image_name"] == image_name].copy()

        normalized_value = records.iloc[0]["normalized"]
        image_root = self.resolve_image_root(normalized_value)
        image_path = os.path.join(image_root, image_name)

        image4 = build_four_channel_image(image_path)
        height, width = image4.shape[:2]

        records.dropna(subset=["xmin", "ymin", "xmax", "ymax"], inplace=True)

        boxes = []
        labels = []

        for _, row in records.iterrows():
            x1, y1, x2, y2 = map(float, row[["xmin", "ymin", "xmax", "ymax"]])

            if str(normalized_value).strip().lower() == "yes":
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height

            boxes.append([x1, y1, x2, y2])
            labels.append(1 if row["class"] == "koala" else 2)

        if self.transforms:
            augmented = self.transforms(image=image4, bboxes=boxes, labels=labels)
            image_tensor = augmented["image"]
            boxes_tensor = torch.tensor(augmented["bboxes"], dtype=torch.float32)
            labels_tensor = torch.tensor(augmented["labels"], dtype=torch.int64)
        else:
            image_tensor = torch.from_numpy(image4.transpose(2, 0, 1) / 255.0).float()
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        return image_tensor, target


def collate_fn(batch: list[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Custom collate function required by torchvision detection models."""
    return tuple(zip(*batch))


# =============================================================================
# Model construction
# =============================================================================

def get_model(num_classes: int) -> torch.nn.Module:
    """Build a Faster R-CNN model modified for 4-channel input.

    The ImageNet-pretrained first convolution expects RGB input. To initialise
    the fourth Sobel channel, the code copies the mean of the RGB convolution
    weights into the extra input channel.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    old_conv = model.backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)

        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    model.backbone.body.conv1 = new_conv

    predictor_input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(predictor_input_features, num_classes)

    original_transform = model.transform
    model.transform = GeneralizedRCNNTransform(
        original_transform.min_size,
        original_transform.max_size,
        image_mean=tuple(original_transform.image_mean) + (0.0,),
        image_std=tuple(original_transform.image_std) + (1.0,),
    )

    return model


def get_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create the SGD optimizer used in the original training pipeline."""
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(
        trainable_parameters,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute validation loss for a torchvision detection model.

    Torchvision Faster R-CNN returns losses only in training mode when targets
    are supplied. Therefore the model is temporarily kept in train mode for
    validation-loss computation.
    """
    model.train()
    total_loss = 0.0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        print(f"  Validating batch {batch_index}/{len(loader)}", end="\r")

        images_on_device = [image.to(device) for image in images]
        targets_on_device = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images_on_device, targets_on_device)
        total_loss += sum(loss_dict.values()).item()

    model.eval()
    print()
    return total_loss / max(1, len(loader))


def train_one_round(
    round_index: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    output_dir: str,
) -> torch.nn.Module:
    """Train one round of the detector and save the checkpoint."""
    print(f"\n=== Training round {round_index} ===")

    model = get_model(num_classes).to(device)
    optimizer = get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    validation_history = []
    early_stop_flags = deque(maxlen=HMA_STOP_PATIENCE)

    epoch = 0
    while True:
        epoch += 1
        print(f"\n--- Round {round_index}, epoch {epoch} ---")

        model.train()
        batch_losses = []

        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images_on_device = [image.to(device) for image in images]
            targets_on_device = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images_on_device, targets_on_device)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            print(
                f"Round {round_index} | Epoch {epoch} | "
                f"Batch {batch_index}/{len(train_loader)} | Loss {loss.item():.4f}",
                end="\r",
            )

        print()
        avg_train_loss = float(np.mean(batch_losses))
        print(f"Round {round_index} | Epoch {epoch} | Avg train loss: {avg_train_loss:.4f}")

        scheduler.step()

        val_loss = validate(model, val_loader, device)
        print(f"Round {round_index} | Epoch {epoch} | Val loss: {val_loss:.4f}")

        validation_history.append(val_loss)
        hma = hull_moving_average(validation_history, HMA_WINDOW)

        if len(validation_history) >= 2 and not np.isnan(hma[-1]) and not np.isnan(hma[-2]):
            slope = hma[-1] - hma[-2]
            stopped_improving = slope > -HMA_SLOPE_THRESHOLD
            early_stop_flags.append(stopped_improving)

            print(
                f"  HMA slope={slope:.6f}; "
                f"stop flags={sum(early_stop_flags)}/{HMA_STOP_PATIENCE}"
            )

        if len(early_stop_flags) == HMA_STOP_PATIENCE and all(early_stop_flags):
            print(f"[EARLY STOP] Round {round_index} stopped at epoch {epoch}")
            break

    checkpoint_path = Path(output_dir) / MODEL_ROUND_TEMPLATE.format(round=round_index)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SAVE] Round {round_index} model saved to {checkpoint_path}")

    return model


# =============================================================================
# Hard-negative mining
# =============================================================================

def init_hard_negative_worker(
    checkpoint_path: str,
    device_name: str,
    images_dir: str,
    alt_images_dir: str,
    orig_annotations: str,
) -> None:
    """Initialise model and annotation data inside each multiprocessing worker."""
    global _WORKER_MODEL, _WORKER_ORIG_DF, _WORKER_CONFIG

    device = torch.device(device_name)

    _WORKER_CONFIG = {
        "device": device,
        "images_dir": images_dir,
        "alt_images_dir": alt_images_dir,
    }

    _WORKER_MODEL = get_model(3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _WORKER_MODEL.load_state_dict(
        checkpoint if not isinstance(checkpoint, dict) else checkpoint.get("model", checkpoint)
    )
    _WORKER_MODEL.eval()

    _WORKER_ORIG_DF = pd.read_csv(orig_annotations).set_index("image_name")


def detect_one_negative_candidate(image_name: str) -> tuple[str, str, list[list[float]]] | None:
    """Run the current model on one negative image and return false positives.

    A negative image becomes a hard negative if the model incorrectly detects a
    koala with confidence above `DETECTION_THRESHOLD`.
    """
    records = _WORKER_ORIG_DF.loc[[image_name]]
    normalized_value = str(records.iloc[0]["normalized"]).strip().lower()

    if normalized_value == "yes":
        root = _WORKER_CONFIG["alt_images_dir"]
    elif normalized_value == "no":
        root = _WORKER_CONFIG["images_dir"]
    else:
        root = normalized_value

    image_path = os.path.join(root, image_name)
    image4 = build_four_channel_image(image_path)
    tensor = torch.from_numpy(image4.transpose(2, 0, 1) / 255.0).float().to(_WORKER_CONFIG["device"])

    with torch.no_grad():
        output = _WORKER_MODEL([tensor])[0]

    mask = (output["labels"] == 1) & (output["scores"] >= DETECTION_THRESHOLD)
    boxes = output["boxes"][mask].cpu().numpy()

    if boxes.size > 0:
        return image_name, normalized_value, boxes.tolist()

    return None


def detect_hard_negatives(
    orig_df: pd.DataFrame,
    positive_count: int,
    round_index: int,
    device_name: str,
    images_dir: str,
    alt_images_dir: str,
    orig_annotations: str,
    output_dir: str,
) -> tuple[int, str, str]:
    """Collect false-positive hard negatives using the previous round's model."""
    has_koala = orig_df.groupby("image_name")["class"].apply(lambda classes: (classes == "koala").any())
    negative_names = [name for name, has_positive in has_koala.items() if not has_positive]
    random.shuffle(negative_names)

    hard_negative_folder = Path(output_dir) / NEG_FOLDER_TEMPLATE.format(round=round_index)
    hard_negative_csv = Path(output_dir) / NEG_CSV_TEMPLATE.format(round=round_index)

    hard_negative_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(output_dir) / MODEL_ROUND_TEMPLATE.format(round=round_index)

    mined_count = 0

    with open(hard_negative_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "class", "xmin", "ymin", "xmax", "ymax", "normalized"])

        with Pool(
            processes=NUM_WORKERS,
            initializer=init_hard_negative_worker,
            initargs=(str(checkpoint_path), device_name, images_dir, alt_images_dir, orig_annotations),
        ) as pool:
            for checked_index, result in enumerate(
                pool.imap_unordered(detect_one_negative_candidate, negative_names),
                start=1,
            ):
                print(
                    f"Mining hard negatives for round {round_index}: "
                    f"{checked_index}/{len(negative_names)} checked | "
                    f"mined {mined_count}/{positive_count}",
                    end="\r",
                )

                if result is None:
                    continue

                image_name, normalized_value, boxes = result
                mined_count += 1

                source_root = alt_images_dir if normalized_value == "yes" else images_dir
                source_path = os.path.join(source_root, image_name)
                destination_path = hard_negative_folder / image_name

                shutil.copy(source_path, destination_path)

                for x1, y1, x2, y2 in boxes:
                    writer.writerow(
                        [
                            image_name,
                            "negative",
                            x1,
                            y1,
                            x2,
                            y2,
                            str(hard_negative_folder),
                        ]
                    )

                if mined_count >= positive_count:
                    break

    print(
        f"\n[INFO] Collected {mined_count} hard negatives in {hard_negative_folder}; "
        f"annotations saved to {hard_negative_csv}"
    )

    return mined_count, str(hard_negative_folder), str(hard_negative_csv)


# =============================================================================
# Dataset assembly
# =============================================================================

def make_data_loaders(
    images_dir: str,
    alt_images_dir: str,
    annotation_csv: str,
    train_names: list[str],
    val_names: list[str],
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders for a selected name split."""
    train_dataset = KoalaDataset(images_dir, alt_images_dir, annotation_csv, get_transform(train=True))
    val_dataset = KoalaDataset(images_dir, alt_images_dir, annotation_csv, get_transform(train=False))

    train_dataset.image_names = train_names
    val_dataset.image_names = val_names

    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() or 1,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() or 1,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def write_round_annotation_csv(
    output_csv: str,
    original_df: pd.DataFrame,
    hard_negative_df: pd.DataFrame,
    positive_names: list[str],
    negative_names: list[str],
    hard_negative_folder: str,
) -> None:
    """Write a balanced positive/negative annotation CSV for one training round."""
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "class", "xmin", "ymin", "xmax", "ymax", "normalized"])

        for image_name in positive_names:
            image_records = original_df[original_df["image_name"] == image_name]
            for _, row in image_records.iterrows():
                if row["class"] == "koala":
                    writer.writerow(
                        [
                            row["image_name"],
                            "koala",
                            row["xmin"],
                            row["ymin"],
                            row["xmax"],
                            row["ymax"],
                            row["normalized"],
                        ]
                    )

        selected_negative_records = hard_negative_df[hard_negative_df["image_name"].isin(negative_names)]
        for _, row in selected_negative_records.iterrows():
            writer.writerow(
                [
                    row["image_name"],
                    "negative",
                    row["xmin"],
                    row["ymin"],
                    row["xmax"],
                    row["ymax"],
                    hard_negative_folder,
                ]
            )


# =============================================================================
# Main pipeline
# =============================================================================

def run_training_pipeline(args: argparse.Namespace) -> None:
    """Run the complete multi-round training and hard-negative mining pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    original_df = pd.read_csv(args.orig_annotations)

    has_koala = original_df.groupby("image_name")["class"].apply(lambda classes: (classes == "koala").any())
    positive_names = [name for name, has_positive in has_koala.items() if has_positive]
    positive_count = len(positive_names)

    print(f"[INFO] Found {positive_count} positive koala images.")

    round_1_model_path = output_dir / MODEL_ROUND_TEMPLATE.format(round=1)

    if not round_1_model_path.exists():
        print("=== Round 1: training on positive koala images only ===")

        positive_df = original_df[original_df["class"] == "koala"]
        pos_annotation_path = output_dir / POS_ANNOTATIONS
        positive_df.to_csv(pos_annotation_path, index=False)

        round_names = sorted(positive_df["image_name"].unique())
        random.shuffle(round_names)

        train_size = int(0.8 * len(round_names))
        train_names = round_names[:train_size]
        val_names = round_names[train_size:]

        train_loader, val_loader = make_data_loaders(
            args.images_dir,
            args.alt_images_dir,
            str(pos_annotation_path),
            train_names,
            val_names,
        )

        model = train_one_round(1, train_loader, val_loader, 3, device, str(output_dir))
    else:
        print(f"[INFO] Loading existing Round 1 model from {round_1_model_path}")
        model = get_model(3).to(device)
        model.load_state_dict(torch.load(round_1_model_path, map_location=device))

    previous_hard_negative_folder = None
    previous_hard_negative_csv = None
    current_round = 1

    while True:
        current_round += 1
        previous_round = current_round - 1

        mined_count, hard_negative_folder, hard_negative_csv = detect_hard_negatives(
            orig_df=original_df,
            positive_count=positive_count,
            round_index=previous_round,
            device_name=args.device,
            images_dir=args.images_dir,
            alt_images_dir=args.alt_images_dir,
            orig_annotations=args.orig_annotations,
            output_dir=str(output_dir),
        )

        if mined_count == 0:
            print("[DONE] No hard negatives found. Training pipeline is complete.")
            break

        pair_count = min(positive_count, mined_count)

        random.shuffle(positive_names)
        selected_positive_names = positive_names[:pair_count]

        hard_negative_df = pd.read_csv(hard_negative_csv)
        selected_negative_names = sorted(hard_negative_df["image_name"].unique())[:pair_count]

        round_annotation_path = output_dir / ANNOTATIONS_ROUND_TEMPLATE.format(round=current_round)

        write_round_annotation_csv(
            str(round_annotation_path),
            original_df,
            hard_negative_df,
            selected_positive_names,
            selected_negative_names,
            hard_negative_folder,
        )

        train_pos_count = int(0.8 * pair_count)

        train_positive = selected_positive_names[:train_pos_count]
        val_positive = selected_positive_names[train_pos_count:]

        train_negative = selected_negative_names[:train_pos_count]
        val_negative = selected_negative_names[train_pos_count:]

        train_names = train_positive + train_negative
        val_names = val_positive + val_negative

        random.shuffle(train_names)
        random.shuffle(val_names)

        train_loader, val_loader = make_data_loaders(
            args.images_dir,
            args.alt_images_dir,
            str(round_annotation_path),
            train_names,
            val_names,
        )

        model = train_one_round(current_round, train_loader, val_loader, 3, device, str(output_dir))

        if previous_hard_negative_folder and previous_hard_negative_csv:
            shutil.rmtree(previous_hard_negative_folder, ignore_errors=True)
            try:
                os.remove(previous_hard_negative_csv)
            except FileNotFoundError:
                pass

        previous_hard_negative_folder = hard_negative_folder
        previous_hard_negative_csv = hard_negative_csv

    print("[DONE] All training rounds complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a 4-channel Faster R-CNN koala detector with hard-negative mining."
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Folder containing original koala images.")
    parser.add_argument("--alt-images-dir", default=DEFAULT_ALT_IMAGES_DIR, help="Folder containing Open Images background images.")
    parser.add_argument("--orig-annotations", default=DEFAULT_ORIG_ANNOTATIONS, help="Original annotation CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder where checkpoints and round CSVs are saved.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device, e.g. cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_training_pipeline(args)


if __name__ == "__main__":
    main()
