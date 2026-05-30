#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
koala_batch_inference.py

Batch inference script for the Koala Be Where detector.

The script loads a trained 4-channel Faster R-CNN model and applies it to all
images and videos inside an input folder.

For every frame:
1. Convert BGR input from OpenCV to RGB.
2. Compute a Sobel edge-magnitude channel.
3. Concatenate RGB + Sobel into a 4-channel tensor.
4. Run Faster R-CNN inference.
5. Draw boxes around koala detections.
6. Save annotated output files.

This script is intended for local testing and visual inspection of the trained
wildlife detector.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform


# =============================================================================
# Default configuration
# =============================================================================

DEFAULT_DEVICE = "cpu"
DEFAULT_NUM_CLASSES = 3
DEFAULT_CHECKPOINT = "batch_checkpoint.pth"
DEFAULT_INPUT_DIR = "/Users/richman/Downloads/untitled folder"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_SCORE_THRESHOLD = 0.75

IMAGE_MEAN4 = (0.485, 0.456, 0.406, 0.0)
IMAGE_STD4 = (0.229, 0.224, 0.225, 1.0)

MIN_SIZE = (800,)
MAX_SIZE = 1333

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# Model object stored once per process.
_MODEL = None
_WORKER_CONFIG = {}


def get_model(num_classes: int) -> torch.nn.Module:
    """Build the same 4-channel Faster R-CNN architecture used in training."""
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

    model.transform = GeneralizedRCNNTransform(
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        image_mean=IMAGE_MEAN4,
        image_std=IMAGE_STD4,
    )

    return model


def load_model(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int,
) -> torch.nn.Module:
    """Load a trained detector checkpoint."""
    model = get_model(num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def init_worker(
    checkpoint_path: str,
    device_name: str,
    num_classes: int,
    score_threshold: float,
) -> None:
    """Initialise model state inside each video-processing worker."""
    global _MODEL, _WORKER_CONFIG

    device = torch.device(device_name)
    _WORKER_CONFIG = {
        "device": device,
        "score_threshold": score_threshold,
    }

    _MODEL = load_model(checkpoint_path, device, num_classes)


def compute_sobel_max(image_rgb: np.ndarray) -> np.ndarray:
    """Compute a Sobel edge-magnitude channel from an RGB frame."""
    sobel_channels = []

    for channel_index in range(3):
        channel = image_rgb[:, :, channel_index]
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_channels.append(magnitude)

    sobel_max = np.max(np.stack(sobel_channels, axis=0), axis=0)
    return (sobel_max / (sobel_max.max() + 1e-6) * 255.0).astype(np.uint8)


def make_four_channel_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert an OpenCV BGR frame into a 4-channel model input tensor."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape

    sobel = compute_sobel_max(frame_rgb).reshape(height, width, 1)
    image4 = np.concatenate([frame_rgb, sobel], axis=2)

    tensor = torch.from_numpy(image4.transpose(2, 0, 1)).float().to(device)
    return tensor / 255.0


def draw_detections(
    frame_bgr: np.ndarray,
    output: dict[str, torch.Tensor],
    score_threshold: float,
) -> np.ndarray:
    """Draw koala detections on one BGR frame."""
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if label != 1 or score < score_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"koala {score * 100:.1f}%",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame_bgr


def process_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Run inference on one video frame.

    This function is multiprocessing-friendly. The model is stored in a process
    global variable initialised by `init_worker`.
    """
    device = _WORKER_CONFIG["device"]
    score_threshold = _WORKER_CONFIG["score_threshold"]

    tensor = make_four_channel_tensor(frame_bgr, device)

    with torch.no_grad():
        output = _MODEL([tensor])[0]

    return draw_detections(frame_bgr, output, score_threshold)


def infer_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    score_threshold: float,
    output_dir: str,
) -> None:
    """Run inference on a single image and save an annotated copy."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    tensor = make_four_channel_tensor(image, device)

    with torch.no_grad():
        output = model([tensor])[0]

    annotated = draw_detections(image, output, score_threshold)

    output_path = Path(output_dir) / f"ann_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)


def infer_video(
    video_path: str,
    checkpoint_path: str,
    device_name: str,
    num_classes: int,
    score_threshold: float,
    output_dir: str,
) -> None:
    """Run frame-by-frame inference on one video and save an annotated video."""
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    capture.release()

    if not frames:
        print(f"[WARN] Video contains no readable frames: {video_path}")
        return

    processed_frames = []

    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(checkpoint_path, device_name, num_classes, score_threshold),
    ) as pool:
        for processed in tqdm(
            pool.imap(process_frame_bgr, frames),
            total=len(frames),
            desc=f"Video {Path(video_path).name}",
        ):
            processed_frames.append(processed)

    output_path = Path(output_dir) / f"ann_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in processed_frames:
        writer.write(frame)

    writer.release()


def discover_inputs(input_dir: str) -> tuple[list[str], list[str]]:
    """Return image and video files found in the input directory."""
    images = []
    videos = []

    for filename in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, filename)

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(filename.lower())[1]

        if extension in IMAGE_EXTENSIONS:
            images.append(path)
        elif extension in VIDEO_EXTENSIONS:
            videos.append(path)

    return images, videos


def run_batch_inference(args: argparse.Namespace) -> None:
    """Run image and video inference over all supported files in a folder."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, videos = discover_inputs(args.input_dir)

    device = torch.device(args.device)

    if images:
        print(f"[INFO] Processing {len(images)} image(s).")
        model = load_model(args.checkpoint, device, args.num_classes)

        for image_path in tqdm(images, desc="Images"):
            infer_image(
                image_path,
                model,
                device,
                args.score_threshold,
                str(output_dir),
            )

    if videos:
        print(f"[INFO] Processing {len(videos)} video(s).")
        for video_path in videos:
            infer_video(
                video_path,
                args.checkpoint,
                args.device,
                args.num_classes,
                args.score_threshold,
                str(output_dir),
            )

    print(f"[DONE] Annotated outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference for the 4-channel Koala Be Where detector."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Folder containing images and videos.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder for annotated outputs.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device, e.g. cpu, cuda, or mps.")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of model classes.")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD, help="Minimum detection confidence.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_batch_inference(args)


if __name__ == "__main__":
    main()
