# Koala Be Where: Four-Channel Koala Detection Pipeline

A computer vision project for detecting koalas in natural environments using a modified Faster R-CNN model, Sobel edge information, Open Images background data, and multi-round hard-negative mining.

This repository demonstrates an end-to-end applied AI workflow:

1. Collecting additional background images
2. Preparing training annotations
3. Training a wildlife detector
4. Extending a pretrained model from RGB input to RGB + edge input
5. Mining hard negatives from false-positive detections
6. Iteratively retraining the model
7. Running batch inference on images and videos
8. Documenting project progress through weekly timesheets

The project was originally developed as part of the **Koala Be Where** project at the University of Adelaide.


---

## Project Structure

```text
koala-be-where-detection/
├── README.md
├── requirements.txt
├── scripts/
│   ├── open_images_downloader.py
│   ├── train_koala_multiround.py
│   └── koala_batch_inference.py
├── docs/
│   └── timesheets/
│       ├── MCI-Timetsheet-week2.xlsx
│       ├── MCI-Timetsheet-week3.xlsx
│       ├── MCI-Timetsheet-week4.xlsx
│       ├── MCI-Timetsheet-week5.xlsx
│       ├── MCI-Timetsheet-week6.xlsx
│       ├── MCI-Timetsheet-week9.xlsx
│       ├── MCI-Timetsheet-week10.xlsx
│       ├── MCI-Timetsheet-week11.xlsx
│       └── MCI-Timetsheet-week12.xlsx
├── data/
│   └── README.md
├── outputs/
│   └── .gitkeep
└── models/
    └── .gitkeep
```

Large datasets and trained model checkpoints should usually not be committed directly to GitHub unless they are small enough and you have permission to publish them. The repository should contain code, documentation, and lightweight project records. Large files can be referenced through external storage if needed.

---

## Main Scripts

### `open_images_downloader.py`

This script downloads additional images from the public Open Images Dataset.

It reads a text file containing image references such as:

```text
train/000002b66c9c498e
validation/0001eeaf4aed83f9
test/0004d51c1ffd0f1c
```

For each entry, the script downloads:

```text
split/image_id.jpg
```

from the public S3 bucket:

```text
open-images-dataset
```

The script uses:

- `boto3` for S3 access
- unsigned public bucket access
- multiprocessing for faster downloading
- `tqdm` progress bars
- automatic skipping of already downloaded files

This was useful for collecting negative/background images for hard-negative mining.

---

### `train_koala_multiround.py`

This is the main training pipeline.

It trains a Faster R-CNN detector with:

- ResNet-50 FPN backbone
- pretrained torchvision weights
- modified 4-channel input layer
- koala positive class
- hard-negative background class
- multi-round training
- false-positive mining
- validation-loss based early stopping

The training process is designed to improve robustness against confusing natural backgrounds such as:

- tree branches
- leaves
- shadows
- rocks
- bark textures
- low-light areas
- koala-like shapes in the environment

---

### `koala_batch_inference.py`

This script runs the trained detector on a folder of images and videos.

It supports common image formats:

```text
.jpg, .jpeg, .png, .bmp
```

and common video formats:

```text
.mp4, .avi, .mov, .mkv
```

For every image or video frame, it:

1. computes the fourth Sobel edge channel,
2. runs model inference,
3. filters detections by confidence score,
4. draws bounding boxes,
5. saves annotated outputs with the prefix `ann_`.

---

## Four-Channel Model Design

A normal pretrained Faster R-CNN model expects 3-channel RGB input:

```text
R, G, B
```

This project modifies the first convolutional layer to accept 4 channels:

```text
R, G, B, Sobel
```

The fourth channel is an edge magnitude map generated from the RGB image.

Conceptually:

```text
RGB image
   ↓
Sobel edge extraction
   ↓
RGB + Sobel channel
   ↓
4-channel Faster R-CNN
   ↓
Koala bounding boxes
```

---

## Sobel Edge Channel

For each RGB channel, the Sobel operator calculates horizontal and vertical gradients:

```text
Sobel_x = gradient in x direction
Sobel_y = gradient in y direction
```

The edge magnitude for one channel is:

```text
magnitude = sqrt(Sobel_x^2 + Sobel_y^2)
```

The script applies this to all three RGB channels and then keeps the maximum edge magnitude at each pixel:

```text
Sobel_max = max(magnitude_R, magnitude_G, magnitude_B)
```

This produces one additional channel:

```text
Sobel_max ∈ [0, 255]
```

The reason for adding this channel is that koalas can be difficult to detect by colour alone. Their shape, outline, and fur boundaries may provide useful information, especially when they are partly hidden by branches or foliage.

---

## First Convolution Modification

The original pretrained ResNet first convolution has weights shaped like:

```text
[out_channels, 3, kernel_height, kernel_width]
```

The modified model creates a new convolution with shape:

```text
[out_channels, 4, kernel_height, kernel_width]
```

The first three channels copy the pretrained RGB weights:

```text
new_weight[:, 0:3] = old_weight
```

The Sobel channel is initialised using the average of the RGB weights:

```text
new_weight[:, 3] = mean(old_weight across RGB channels)
```

This preserves pretrained knowledge while giving the model a reasonable starting point for the new edge channel.

---

## Classes

The detector uses three class indices:

```text
0 = background
1 = koala
2 = negative / hard-negative object
```

Class 0 is handled internally by Faster R-CNN. The annotation CSVs use class labels such as:

```text
koala
negative
```

The dataset class converts them to numeric labels.

---

## Annotation Format

The training CSV should contain:

```text
image_name,class,xmin,ymin,xmax,ymax,normalized
```

Example:

```text
koala_001.jpg,koala,120,80,320,360,no
background_001.jpg,negative,40,70,180,240,hard_neg_round2
```

The `normalized` field controls how the image is loaded and how coordinates are interpreted:

| Value | Meaning |
|---|---|
| `yes` | Image is from Open Images; box coordinates are normalised |
| `no` | Image is from the original koala archive; box coordinates are absolute |
| folder name | Image is from a mined hard-negative folder |

---

## Multi-Round Hard-Negative Mining

Hard-negative mining is the key idea behind the training pipeline.

The detector may initially make false-positive predictions on background images. Instead of ignoring those mistakes, the pipeline collects them and uses them as negative training examples in the next round.

The loop is:

```text
Train detector
      ↓
Run detector on negative images
      ↓
Find false positives
      ↓
Save false-positive boxes as hard negatives
      ↓
Combine koala positives + hard negatives
      ↓
Train next round
```

This makes the detector progressively better at separating true koalas from visually confusing backgrounds.

---

## Balanced Positive / Negative Training

Each mining round creates a balanced dataset:

```text
number of positive koala images ≈ number of hard-negative images
```

This prevents the detector from being overwhelmed by one class and gives it a clearer learning signal.

---

## Early Stopping with Hull Moving Average

The training loop uses validation loss and a Hull Moving Average to decide when to stop a round.

The Hull Moving Average is:

```text
HMA(n) = WMA(2 × WMA(loss, n/2) - WMA(loss, n), sqrt(n))
```

Where `WMA` means weighted moving average.

The idea is to smooth noisy validation loss and detect when improvement has slowed down.

If the smoothed validation loss stops improving for several checks, the round stops automatically.

---

## Inference Pipeline

For each image or video frame:

```text
Input frame
   ↓
Convert BGR to RGB
   ↓
Compute Sobel channel
   ↓
Create 4-channel tensor
   ↓
Run Faster R-CNN
   ↓
Keep koala detections above threshold
   ↓
Draw bounding boxes
   ↓
Save annotated result
```

A confidence threshold is used:

```text
score >= SCORE_THRESHOLD
```

For example:

```text
SCORE_THRESHOLD = 0.75
```

Only koala detections above this threshold are drawn.

---

## Installation

Install dependencies:

```bash
pip install torch torchvision pandas numpy pillow opencv-python albumentations tqdm boto3 botocore
```

Depending on your machine, you may want to install a PyTorch build that supports CUDA or Apple Silicon MPS.

---

## Example Usage

### 1. Download Open Images background images

```bash
python scripts/open_images_downloader.py \
  --image-list-file /path/to/image_list.txt \
  --download-folder /path/to/openimages \
  --num-processes 8
```

---

### 2. Train the multi-round detector

```bash
python scripts/train_koala_multiround.py \
  --images-dir /path/to/Koalas \
  --alt-images-dir /path/to/openimages \
  --orig-annotations /path/to/annotations_updated.csv \
  --output-dir outputs/training \
  --device cpu
```

Use `cuda` or `mps` if available:

```bash
python scripts/train_koala_multiround.py --device cuda
```

or:

```bash
python scripts/train_koala_multiround.py --device mps
```

---

### 3. Run batch inference

```bash
python scripts/koala_batch_inference.py \
  --checkpoint outputs/training/model_round3.pth \
  --input-dir /path/to/test_images_or_videos \
  --output-dir outputs/inference \
  --score-threshold 0.75
```

---

## Timesheets

The repository can include weekly project timesheets in:

```text
docs/timesheets/
```

These timesheets are useful as supporting documentation because they show project planning, weekly progress, implementation records, and development activity across multiple weeks.

Suggested files:

```text
MCI-Timetsheet-week2.xlsx
MCI-Timetsheet-week3.xlsx
MCI-Timetsheet-week4.xlsx
MCI-Timetsheet-week5.xlsx
MCI-Timetsheet-week6.xlsx
MCI-Timetsheet-week9.xlsx
MCI-Timetsheet-week10.xlsx
MCI-Timetsheet-week11.xlsx
MCI-Timetsheet-week12.xlsx
```

If the timesheets contain private information, supervisor comments, student IDs, or internal university details, review and clean them before publishing the repository.

---

## Practical Notes

This project contains research and prototype code rather than a packaged production system.

Before running on another machine, update paths such as:

```text
images directory
Open Images directory
annotation CSV path
checkpoint path
input folder
output folder
```

The scripts now support command-line arguments, so local absolute paths can be overridden without editing the code directly.

---

## Limitations

1. The detector depends heavily on annotation quality.
2. Hard-negative mining can reinforce mistakes if false positives are not reviewed.
3. Large datasets and model checkpoints may be too large for GitHub.
4. Video inference can be memory-heavy because frames are loaded before processing.
5. The default device is CPU, which is slow for deep learning training.
6. The project is designed for experimentation and portfolio demonstration, not production deployment.

---

## Portfolio Value

This repository demonstrates:

- Python engineering
- Computer vision model training
- PyTorch and torchvision detection models
- Transfer learning
- Custom model architecture modification
- 4-channel image preprocessing
- Sobel edge feature engineering
- Dataset construction
- Annotation handling
- Hard-negative mining
- Multiprocessing
- Image and video inference
- Experiment documentation

It is a strong example of turning an academic computer vision idea into a structured, reproducible engineering workflow.

---

## Disclaimer

This repository is for educational, research, and portfolio demonstration purposes only.

The Open Images Dataset and any external datasets should be used according to their respective licences and terms.

The detector output should be manually validated before being used in any real conservation, research, or operational context.
