# Distillation for DL Frameworks Class

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Run](#run)
  - [Faster R-CNN Distillation](#faster-r-cnn-distillation)
    - [Environment Setup](#environment-setup)
    - [Training](#training)
    - [Testing](#testing)
  - [YOLO Distillation Based on MMDetection](#yolo-distillation-based-on-mmdetection)
    - [MMDetection Environment Setup](#mmdetection-environment-setup)
    - [Download Teacher Model Weights](#download-teacher-model-weights)
    - [Model Distillation](#model-distillation)
      - [Training](#training-1)
      - [Testing](#testing-1)
  - [Test Results](test-results)

## Introduction

This project involves distillation for Faster R-CNN and YOLOX based on MMDetection. The dataset used is the COCO dataset. The directory structure is as follows:

```
├── checkpoint                  # Stores weight files
│   ├── faster_rcnn_distill
│   └── yolox_distill
├── data                        # Dataset
│   └── coco2017
├── faster_rcnn_distill         # Distillation code for Faster R-CNN
│   ├── coco2017
│   └── models
├── mmdetection_distill_yolo    # Distillation code for YOLOX
│   ├── distill
│   └── mmdetection
└── README.md
```

The project contains two main folders: `faster_rcnn_distill` for Faster R-CNN distillation and `mmdetection_distill_yolo` for YOLOX distillation.

- [ ] Test results
- [x] Training guide
- [ ] Upload weights to public path

## Data Preparation

To prepare the COCO 2017 dataset, follow these steps:

```bash
cd data/coco2017
# Download COCO 2017 training and validation sets
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# Unzip datasets
unzip -q train2017.zip -d .
unzip -q val2017.zip -d .
unzip -q annotations_trainval2017.zip -d .
```

## Run

Clone the repository:

```bash
git clone https://github.com/BaoZi-chu/distill_for_DLFrameworks_class.git
```

### Faster R-CNN Distillation 

#### Environment Setup

Set up the environment for Faster R-CNN:

```bash
conda install pytorch torchvision -c pytorch
```
The required torch version is different from that based on mmdetection.
#### Training

To train the Faster R-CNN model, run:

```bash
python faster_rcnn_distill/coco2017/train.py
```

#### Testing

To test the Faster R-CNN model, run:

```bash
python faster_rcnn_distill/coco2017/test.py
```

### YOLO Distillation Based on MMDetection

Using MMDetection to implement distillation for YOLOX on the COCO dataset.

#### MMDetection Environment Setup

Set up the environment for MMDetection:

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# Check available torch version from https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html
# Install corresponding torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4, <2.2.0"
# mmcv2.2.0 causes issues with mmdet installation, install version 2.1 instead.
mim install mmdet

cd faster_rcnn_distill
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

#### Download Teacher Model Weights

Download the teacher model weights:

```bash
# Go back to the root directory
cd ../../
wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth -P checkpoint/yolox_distill/
```

#### Model Distillation

##### Training

To train the YOLOX model, run:

```bash
python mmdetection_distill_yolo/distill/distill_train.py
```

##### Testing

To test the YOLOX model, run:

```bash
python mmdetection_distill_yolo/distill/test.py
```
## Trained Weights
https://1drv.ms/f/c/3c1547dca7a7eaf5/EmDyuLbw4INKsjRZLfJWKXYBuyCA2jB4Vxk_8I6qXm4RxA?e=Y99xLo
## Test Results

Below are the test results for both the teacher and student models for Faster R-CNN and YOLOX, presented in a table format:

The random seed of the test is 0. Due to the computational power limitation, the distillation effect is average at present. Among them, Faster R-CNN shrinks by 30% and yolox shrinks by 25%.

| Metric                                | Faster R-CNN Teacher | Faster R-CNN Student | YOLOX Teacher | YOLOX Student |
|---------------------------------------|----------------------|----------------------|---------------|---------------|
| Average Precision (AP) @ IoU=0.50:0.95 | 0.403                | 0.237                | 0.370         | 0.152         |
| Average Precision (AP) @ IoU=0.50      | 0.591                | 0.421                | 0.585         | 0.284         |
| Average Precision (AP) @ IoU=0.75      | 0.434                | 0.238                | 0.398         | 0.149         |
| Average Precision (AP) @ IoU=0.50:0.95 (small) | 0.235                | 0.135                | 0.212         | 0.068         |
| Average Precision (AP) @ IoU=0.50:0.95 (medium) | 0.445                | 0.255                | 0.404         | 0.171         |
| Average Precision (AP) @ IoU=0.50:0.95 (large)  | 0.532                | 0.309                | 0.482         | 0.195         |
| Average Recall (AR) @ IoU=0.50:0.95   | 0.549                | 0.240                | 0.500         | 0.286         |
| Average Recall (AR) @ IoU=0.50:0.95 (1 detection) | 0.491                | 0.398                | 0.308         | 0.286         |
| Average Recall (AR) @ IoU=0.50:0.95 (10 detections) | 0.601                | 0.418                | 0.485         | 0.171         |
| Average Recall (AR) @ IoU=0.50:0.95 (small) | 0.346                | 0.234                | 0.316         | 0.119         |
| Average Recall (AR) @ IoU=0.50:0.95 (medium) | 0.549                | 0.452                | 0.544         | 0.322         |
| Average Recall (AR) @ IoU=0.50:0.95 (large)  | 0.693                | 0.538                | 0.649         | 0.347         |

