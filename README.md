# Ultralytics Computer Vision Project

This project demonstrates advanced computer vision techniques using the Ultralytics library, including SAM2 for video segmentation and YOLOE for object detection.

## Project Structure

```
ultralytics_test/
├── models/                 # Model weights directory
│   ├── sam2.1_t.pt         # SAM2 Tiny model
│   ├── yoloe-26n-seg.pt    # YOLOE Nano segmentation model
│   └── yolo26n-seg.pt      # YOLOv26 Nano segmentation model
├── test_data/              # Test videos and images
├── SAM2_bboxes_prompt.py   # SAM2 video tracker with bounding box prompts
├── yoloe_box_prompt.py     # YOLOE with box prompts
├── yoloe_text_prompt.py    # YOLOE with text prompts
├── test_ultralytics.py     # Test script for ultralytics functionality
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download model weights to the `models/` directory (see models/README.md for instructions)

## Usage

### SAM2 Video Tracker
```bash
python SAM2_bboxes_prompt.py
```
This opens a GUI where you can select regions of interest in a video and track them using SAM2.

### YOLOE Box Prompt Detection
```bash
python yoloe_box_prompt.py
```
This allows you to select regions in a video and detect objects using YOLOE with box prompts.

### YOLOE Text Prompt Detection
```bash
python yoloe_text_prompt.py
```
This performs object detection based on text descriptions.

### Test Script
```bash
python test_ultralytics.py
```
This tests the ultralytics installation and basic functionality.

## Requirements

- Python >= 3.8
- See `requirements.txt` for detailed dependencies
- CUDA-compatible GPU (recommended for optimal performance)

## Notes

- Model files are stored in the `models/` directory
- Test videos should be placed in the `test_data/` directory
- All model paths in the code are configured to look in the `models/` directory