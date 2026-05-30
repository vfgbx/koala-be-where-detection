#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
open_images_downloader.py

Parallel image downloader for the Open Images Dataset.

This script reads a text file containing Open Images image references such as:

    train/000002b66c9c498e
    validation/0001eeaf4aed83f9
    test/0004d51c1ffd0f1c

For each valid entry, it downloads the corresponding JPEG file from the public
Open Images S3 bucket.

The script is designed for dataset preparation in the Koala Be Where computer
vision project, where additional non-koala / background images were required for
hard-negative mining and robustness testing.

Notes:
- The Open Images bucket is public, so unsigned S3 access is used.
- Existing files are skipped automatically.
- Multiprocessing is used to speed up large download lists.
- Keep request rates reasonable and use this script responsibly.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from concurrent import futures
from pathlib import Path
from typing import Generator, Iterable, Tuple

import boto3
import botocore
import tqdm


# =============================================================================
# Default configuration
# =============================================================================

DEFAULT_IMAGE_LIST_FILE = "/Users/richman/image_list.txt"
DEFAULT_DOWNLOAD_FOLDER = "/Users/richman/Downloads/openimages"
DEFAULT_NUM_PROCESSES = multiprocessing.cpu_count()

BUCKET_NAME = "open-images-dataset"

# Open Images object keys usually look like:
#   train/<hex_image_id>.jpg
#   validation/<hex_image_id>.jpg
#   test/<hex_image_id>.jpg
#   challenge2018/<hex_image_id>.jpg
OPEN_IMAGES_KEY_REGEX = r"(test|train|validation|challenge2018)/([a-fA-F0-9]+)"


# Each child process initialises its own S3 bucket object.
# This avoids repeatedly constructing boto3 resources for every single image.
_BUCKET = None


def parse_open_images_entry(image_reference: str) -> Tuple[str, str]:
    """Parse one Open Images reference into `(split, image_id)`.

    Parameters
    ----------
    image_reference:
        A string such as `train/abc123`, `train/abc123.jpg`, or a longer path
        containing an Open Images split and a hexadecimal image id.

    Returns
    -------
    tuple[str, str]
        The dataset split and image id.

    Raises
    ------
    ValueError
        If the line does not contain a valid Open Images reference.
    """
    cleaned = image_reference.strip().replace(".jpg", "")
    match = re.match(OPEN_IMAGES_KEY_REGEX, cleaned)

    if not match:
        raise ValueError(f"Unrecognised Open Images entry: {image_reference!r}")

    split, image_id = match.groups()
    return split, image_id


def iter_clean_image_list(filepath: str) -> Generator[Tuple[str, str], None, None]:
    """Yield validated `(split, image_id)` pairs from a text file.

    The function fails fast if an invalid line is found. This is intentional:
    for dataset preparation, it is usually better to fix the source list rather
    than silently skip entries and create an incomplete dataset.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                yield parse_open_images_entry(raw_line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid image reference at line {line_no}: {raw_line.strip()!r}"
                ) from exc


def worker_init() -> None:
    """Initialise the S3 bucket object inside each child process."""
    global _BUCKET

    _BUCKET = boto3.resource(
        "s3",
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    ).Bucket(BUCKET_NAME)


def download_one_image(task: Tuple[str, str, str]) -> bool:
    """Download one image from the Open Images S3 bucket.

    Parameters
    ----------
    task:
        A tuple containing `(split, image_id, download_folder)`.

    Returns
    -------
    bool
        True if the file exists after the function returns. This includes both
        newly downloaded files and files that already existed.
    """
    split, image_id, download_folder = task
    save_path = os.path.join(download_folder, f"{image_id}.jpg")

    if os.path.exists(save_path):
        return True

    object_key = f"{split}/{image_id}.jpg"

    try:
        _BUCKET.download_file(object_key, save_path)
        return True
    except botocore.exceptions.ClientError as exc:
        print(f"[ERROR] Failed to download {object_key}: {exc}")
        return False


def build_download_tasks(
    image_list_file: str,
    download_folder: str,
) -> list[Tuple[str, str, str]]:
    """Read the image list and convert it into multiprocessing tasks."""
    return [
        (split, image_id, download_folder)
        for split, image_id in iter_clean_image_list(image_list_file)
    ]


def download_all_images(
    image_list_file: str,
    download_folder: str,
    num_processes: int,
) -> None:
    """Download all images listed in `image_list_file`.

    The output directory is created automatically. Existing files are skipped, so
    the downloader can be safely re-run after interruption.
    """
    Path(download_folder).mkdir(parents=True, exist_ok=True)
    tasks = build_download_tasks(image_list_file, download_folder)

    total = len(tasks)
    if total == 0:
        print("[INFO] No valid image references found.")
        return

    print(f"[INFO] Preparing to download {total} images to: {download_folder}")
    print(f"[INFO] Worker processes: {num_processes}")

    completed = 0
    failed = 0

    with tqdm.tqdm(total=total, desc="Downloading", leave=True) as progress:
        with futures.ProcessPoolExecutor(
            max_workers=num_processes,
            initializer=worker_init,
        ) as executor:
            for ok in executor.map(download_one_image, tasks):
                completed += int(ok)
                failed += int(not ok)
                progress.update(1)

    print(f"[DONE] Download finished. Available files: {completed}; failed: {failed}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel downloader for Open Images image ids."
    )
    parser.add_argument(
        "--image-list-file",
        default=DEFAULT_IMAGE_LIST_FILE,
        help="Path to a text file containing Open Images references.",
    )
    parser.add_argument(
        "--download-folder",
        default=DEFAULT_DOWNLOAD_FOLDER,
        help="Directory where downloaded JPEG files will be saved.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=DEFAULT_NUM_PROCESSES,
        help="Number of worker processes. Defaults to all CPU cores.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    try:
        download_all_images(
            image_list_file=args.image_list_file,
            download_folder=args.download_folder,
            num_processes=max(1, args.num_processes),
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting safely.")


if __name__ == "__main__":
    main()
