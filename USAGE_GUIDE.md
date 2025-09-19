# YOLOv12 Ear Detection Finetuning Guide

## Quick Start

1. **Prepare the dataset:**
   ```bash
   python prepare_dataset.py
   ```

2. **Start finetuning:**
   ```bash
   python finetune_ear_model.py
   ```

3. **Evaluate the results:**
   ```bash
   python evaluate_model.py runs/finetune/ear_finetune_XXXXXX/weights/best.pt --original best.pt
   ```

## Files Created

- `prepare_dataset.py` - Organizes AWE and CVLE datasets into YOLO format
- `finetune_ear_model.py` - Main finetuning script
- `evaluate_model.py` - Evaluation and comparison script
- `requirements.txt` - Dependencies (skip torch installation)

## Dataset Structure After Preparation

```
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## Custom Training Parameters

```bash
python finetune_ear_model.py --epochs 50 --batch 8 --lr0 0.0001
```

