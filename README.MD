# Hybrid Model for Deepfake Detection

A research-oriented hybrid deepfake detection system that combines complementary visual representations from multiple deep learning branches.

> **Project status:** Research / Experimental Implementation

## Overview

This project investigates a **hybrid deepfake detection architecture** designed to combine different types of visual evidence instead of relying on a single detector.

The implementation is developed from the ideas of the paper:

**A Hybrid Model for Generalizable Deepfake Detection via Blending, Semantic, and General Artifacts**

and takes selected data-processing and evaluation concepts from:

**DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection**

The project is an **independent implementation and experimental system**. It is not the official implementation of either the referenced paper or DeepfakeBench.

---

## Main Idea

The system is implemented as a two-stage pipeline.

### Stage 1 — BAD detector

Stage 1 trains a **MaxViT-Tiny 384** detector using **Self-Blended Images (SBI)** generated from real face frames.

The purpose of this stage is to obtain a detector/checkpoint that learns artifacts related to face blending and manipulation boundaries.

```text
Real Face Frame
      │
      ▼
SBI Online Generation
      │
      ├───────────────┐
      ▼               ▼
   Fake Image      Real Image
      │               │
      └───────┬───────┘
              ▼
       MaxViT-Tiny 384
              │
              ▼
       Binary Detector
       Real / Deepfake
```

The Stage 1 implementation contains an SBI data generator and a MaxViT-Tiny 384 classifier.

The resulting BAD checkpoint is later used as a **frozen feature extractor** in Stage 2.

---

## Stage 2 — Hybrid Detector

Stage 2 combines three feature branches:

| Branch | Model | Role | Feature dimension | Training in Stage 2 |
|---|---|---|---:|---|
| BAD | MaxViT-Tiny 384 | Blending/artifact representation | 512 | Frozen |
| SFE | SigLIP2 So400M Patch14 384 | Semantic/high-level representation | 1152 | Frozen |
| GAD | ConvNeXt V2-Tiny | General visual/artifact representation | 768 | Trainable |

The three feature vectors are projected into a common **512-dimensional feature space** and concatenated.

```text
                         Input Face
                             │
                Shared Augmentation / Preprocess
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    BAD Branch          SFE Branch          GAD Branch
 MaxViT-Tiny 384      SigLIP2 So400M      ConvNeXt V2-Tiny
    Frozen               Frozen             Trainable
      │                    │                   │
      ▼                    ▼                   ▼
   512-dim              1152-dim             768-dim
      │                    │                   │
      ▼                    ▼                   ▼
 Projection            Projection           Projection
      │                    │                   │
      └──────────────┬─────┴──────┬────────────┘
                     ▼
              Feature Concatenation
                     │
                     ▼
                Fusion Classifier
                     │
                     ▼
                Real / Deepfake
```

### Feature fusion

Each branch is projected using:

```text
Linear
  ↓
LayerNorm
  ↓
GELU
  ↓
Dropout
  ↓
512-dimensional representation
```

The three projected representations are concatenated:

```text
BAD  : 512
SFE  : 512
GAD  : 512
----------------
Fusion: 1536
```

The fusion classifier then produces the final two-class prediction.

---

## Auxiliary Supervision

Stage 2 does not rely only on the final fusion classifier.

Each branch also has its own auxiliary classifier:

```text
BAD projection ──► BAD auxiliary classifier
SFE projection ──► SFE auxiliary classifier
GAD projection ──► GAD auxiliary classifier
```

During training, the loss contains:

```text
Fusion Loss
+
Auxiliary BAD Loss
+
Auxiliary SFE Loss
+
Auxiliary GAD Loss
```

The auxiliary loss is weighted by `aux_weight` in the configuration.

This encourages each branch to retain discriminative information rather than allowing the fusion classifier to depend almost entirely on one branch.

During evaluation, the implementation combines the four logits:

```text
Final Logits =
    (Fusion
     + BAD Auxiliary
     + SFE Auxiliary
     + GAD Auxiliary) / 4
```

The final fake probability is obtained from the softmax output.

---

# Project Structure

```text
Hybird_Model_for_DF/
│
├── stage1_maxvit/
│   ├── config/
│   │   └── stage1_maxvit.yaml
│   │
│   ├── data/
│   │   └── FaceForensics++/
│   │
│   ├── datasets/
│   │   ├── stage1_sbi_dataset.py
│   │   └── ffpp_evaluation_dataset.py
│   │
│   ├── preprocessing/
│   │   ├── preprocess.py
│   │   ├── rearrange.py
│   │   └── dlib_tools/
│   │
│   ├── sbi/
│   │   ├── bi_online_generation.py
│   │   ├── DeepFakeMask.py
│   │   └── sbi_api.py
│   │
│   ├── stage1_models/
│   │   └── maxvit.py
│   │
│   ├── optimizers/
│   │   ├── sam.py
│   │   ├── warmup_cosine_lr.py
│   │   └── linear_lr.py
│   │
│   ├── tools/
│   │   ├── check_dataset.py
│   │   ├── exportsample.py
│   │   └── inspect_checkpoint.py
│   │
│   └── train.py
│
├── stage2v4/
│   ├── configs/
│   │   ├── stage2.yaml
│   │   ├── cross_test.yaml
│   │   └── cross_test_deepfacelab.yaml
│   │
│   ├── dataset_json/
│   │   ├── FaceForensics++.json
│   │   ├── Celeb-DF-v2.json
│   │   ├── DFDC.json
│   │   ├── DFDCP.json
│   │   ├── DeepFakeDetection.json
│   │   ├── FaceShifter.json
│   │   ├── UADFV.json
│   │   └── deepfacelab.json
│   │
│   ├── hybrid_models/
│   │   ├── bad_maxvit_tiny.py
│   │   ├── siglip2_sfe.py
│   │   ├── convnextv2_gad.py
│   │   └── stage2_hybrid.py
│   │
│   ├── stage2_data/
│   │   ├── deepfakebench_dataset.py
│   │   └── transforms.py
│   │
│   ├── stage2_utils/
│   │   ├── balanced_source_sampler.py
│   │   ├── build.py
│   │   ├── common.py
│   │   └── metrics.py
│   │
│   ├── Grad_CAM++/
│   ├── histogram_prediction/
│   ├── test/
│   │
│   ├── train_stage2.py
│   └── test_cross_dataset.py
│
└── README.md
```

---

# Dataset

The main training configuration uses **FaceForensics++ (FF++)**, compression level **c23**.

The configured manipulation sources are:

```text
FF-real
FF-DF     → DeepFakes
FF-F2F    → Face2Face
FF-FS     → FaceSwap
FF-NT     → NeuralTextures
```

The Stage 2 configuration uses up to **8 frames per video** for training and validation.

The configured source-balanced targets are:

```text
Training:
  Real: 5746 frames
  Fake: 5746 frames

Validation:
  Real: 1118 frames
  Fake: 1118 frames
```

The fake samples are distributed across the configured fake manipulation sources.

The repository also contains JSON configurations for additional datasets, including:

- Celeb-DF-v2
- DeepFakeDetection
- DFDC
- DFDCP
- FaceShifter
- UADFV
- DeepFaceLab

These configurations support cross-dataset evaluation experiments.

> Dataset files and videos are not included in this repository. Users must obtain datasets from their respective official sources and comply with their licenses/terms.

---

# Preprocessing

The project follows a face-frame based pipeline.

For Stage 1, the repository includes preprocessing utilities based on:

- Face detection
- Face alignment / landmarks
- Face cropping
- Frame sampling
- SBI online generation

The repository includes the Dlib landmark predictor required by the preprocessing workflow.

For Stage 2, the project uses a shared RGB image and applies synchronized augmentation before generating branch-specific inputs.

The training augmentation includes:

- Horizontal flip
- Rotation
- Gaussian blur
- Random brightness/contrast
- Fancy PCA
- Hue/saturation variation
- Image compression

The same augmented source image is then transformed according to the requirements of each branch.

---

# Stage 1 Training

Stage 1 uses:

```text
Model:
    MaxViT-Tiny 384

Input:
    384 × 384 RGB face image

Training data:
    FaceForensics++ source frames
    + online Self-Blended Image generation

Loss:
    Cross Entropy

Optimizer:
    SAM (Sharpness-Aware Minimization)

Scheduler:
    Warmup + Cosine schedule
```

The default configuration is located at:

```text
stage1_maxvit/config/stage1_maxvit.yaml
```

Example:

```bash
cd stage1_maxvit
python train.py --config config/stage1_maxvit.yaml
```

Stage 1 training requires CUDA according to the current project configuration.

The trained checkpoint is subsequently used as the BAD branch in Stage 2.

---

# Stage 2 Training

The default Stage 2 configuration is:

```text
stage2v4/configs/stage2.yaml
```

Important settings include:

```yaml
data:
  compression: c23
  max_frames_per_video_train: 8
  max_frames_per_video_val: 8
  image_size: 384

bad:
  model_name: maxvit_tiny_tf_384.in1k
  feature_dim: 512

sfe:
  model_name: google/siglip2-so400m-patch14-384
  feature_dim: 1152

gad:
  feature_dim: 768

fusion:
  projection_dim: 512
  num_classes: 2
```

Default training settings:

```yaml
train:
  epochs: 25
  batch_size: 16
  gad_lr: 0.00003
  head_lr: 0.0001
  weight_decay: 0.0005
  warmup_epochs: 1
  aux_weight: 0.3
  gradient_clip_norm: 1.0
```

Run Stage 2 training:

```bash
cd stage2v4
python train_stage2.py --config configs/stage2.yaml
```

The exact command-line arguments should be checked with:

```bash
python train_stage2.py --help
```

---

# Frozen and Trainable Components

A key design decision in Stage 2 is to avoid updating all three large feature extractors simultaneously.

### Frozen

```text
BAD
└── MaxViT-Tiny 384

SFE
└── SigLIP2 So400M Patch14 384
```

### Trainable

```text
GAD
└── ConvNeXt V2-Tiny

Fusion projections
├── BAD projection
├── SFE projection
└── GAD projection

Auxiliary classifiers
├── BAD auxiliary classifier
├── SFE auxiliary classifier
└── GAD auxiliary classifier

Fusion classifier
└── final classifier
```

This reduces the number of parameters that need to be updated during Stage 2 training while preserving multiple pretrained representations.

---

# Cross-Dataset Evaluation

The project contains a dedicated cross-dataset evaluation script:

```text
stage2v4/test_cross_dataset.py
```

The evaluation pipeline calculates both frame-level and video-level metrics.

Example configuration files:

```text
stage2v4/configs/cross_test.yaml
stage2v4/configs/cross_test_deepfacelab.yaml
```

The evaluation process aggregates frame predictions belonging to the same video by taking the mean score.

Conceptually:

```text
Frame 1 ─┐
Frame 2 ─┤
Frame 3 ─┤──► Mean score ──► Video prediction
  ...    ─┤
Frame N ─┘
```

This allows the system to report both:

- Frame-level performance
- Video-level performance

---

# Evaluation Metrics

The implementation calculates:

- Accuracy
- Balanced Accuracy
- Real-class Accuracy
- Fake-class Accuracy
- ROC-AUC
- EER
- Average Precision (AP)
- Precision
- Recall
- Loss

For cross-dataset testing, the results are reported separately at frame and video levels.

The evaluation design follows the general direction of standardized deepfake evaluation used by DeepfakeBench, which reports frame-level AUC, video-level AUC, accuracy, EER, precision-recall and AP.

---

# Explainability

The repository also contains experimental **Grad-CAM++** utilities:

```text
stage2v4/Grad_CAM++/
```

These scripts are intended to inspect which image regions contribute to the model's predictions.

The purpose is to provide qualitative analysis in addition to numerical classification metrics.

Example analysis workflow:

```text
Input Face
    │
    ▼
Hybrid Model
    │
    ▼
Prediction
    │
    ▼
Grad-CAM++
    │
    ▼
Activation / Attention Visualization
```

This can be useful for investigating whether the detector focuses on facial boundaries, blending regions, texture inconsistencies, or other visual artifacts.

---

# Prediction Distribution Analysis

The repository also contains:

```text
stage2v4/histogram_prediction/
```

These utilities can be used to analyze the distribution of prediction scores.

For example:

```text
Probability
0.0 ─────────────────────────────── 1.0
 │
 ├── Real samples
 │
 └── Fake samples
```

This provides an additional view of class separation beyond a single accuracy or AUC value.

---

# Dependencies

The project uses Python and PyTorch together with several computer-vision and deep-learning libraries.

Important dependencies include:

```text
Python
PyTorch
torchvision
timm
transformers
albumentations
Pillow
NumPy
scikit-learn
PyYAML
tqdm
```

The Stage 2 SFE branch uses Hugging Face Transformers to load:

```text
google/siglip2-so400m-patch14-384
```

The BAD branch uses `timm`:

```text
maxvit_tiny_tf_384.in1k
```

The GAD branch contains a local implementation of the required dense 2D ConvNeXt V2-Tiny architecture so that the project does not need the sparse-convolution components of the original ConvNeXt V2 repository.

---

# Hardware

Deepfake detection with multiple high-capacity vision models is computationally expensive.

A CUDA-enabled NVIDIA GPU is recommended for training.

The current Stage 1 configuration explicitly requires CUDA.

Stage 2 uses large pretrained visual encoders, including MaxViT-Tiny and SigLIP2 So400M, so full training is not intended to be lightweight CPU training.

CPU execution may be useful for:

- Debugging
- Dataset inspection
- Small inference tests
- Code development
- Checkpoint inspection

For full experiments, a GPU environment such as Kaggle or Google Colab can be used when sufficient GPU memory is available.

---

# Reproducibility

The project sets a deterministic/random seed through the configuration:

```yaml
seed: 5
```

The training pipeline also includes:

- Worker initialization
- Configurable frame sampling
- Fixed dataset JSON files
- Explicit checkpoint paths
- Configurable learning rates
- Warmup scheduling
- Gradient clipping
- Checkpoint retention

Nevertheless, exact reproducibility can still depend on:

- CUDA version
- PyTorch version
- GPU architecture
- Dataset preprocessing
- Pretrained checkpoint versions
- Randomness in data augmentation

---

# Research Relationship

The project is conceptually inspired by the idea of combining complementary artifact representations for more generalizable deepfake detection.

The implementation separates the representation into three practical branches:

```text
BAD
Blending / manipulation-related artifacts
        +
SFE
Semantic / high-level visual representation
        +
GAD
General visual artifact representation
        ↓
     Feature Fusion
        ↓
   Deepfake Detection
```

This should be understood as an **experimental adaptation of the research idea**, rather than a claim that the implementation exactly reproduces the original paper's architecture or experimental setup.

---

# Relation to DeepfakeBench

DeepfakeBench is used as a reference for several aspects of the project, particularly:

- Dataset organization
- Dataset JSON representation
- Face-frame based processing
- Standardized train/validation/test handling
- Cross-dataset evaluation
- Frame-level and video-level metrics
- General deepfake detection experimentation workflow

The project does **not** simply copy the entire DeepfakeBench framework.

Instead, selected ideas and implementation patterns were adapted into a custom hybrid architecture.

DeepfakeBench itself is a comprehensive benchmark that provides unified data management, detector implementations and standardized evaluation protocols for deepfake detection.

---

# References

## Main Research Paper

**A Hybrid Model for Generalizable Deepfake Detection via Blending, Semantic, and General Artifacts**

ACM Digital Library:

https://dl.acm.org/doi/10.1145/3709020.3734833

DOI:

```text
10.1145/3709020.3734833
```

---

## DeepfakeBench

**DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection**

GitHub:

https://github.com/SCLBD/DeepfakeBench

The project uses DeepfakeBench as a reference for dataset organization, preprocessing concepts and evaluation methodology.

---

## Self-Blended Images

The Stage 1 data-generation process is based on the Self-Blended Images (SBI) approach.

The project uses SBI-style online generation to construct training samples from real face images.

DeepfakeBench includes SBI among its supported deepfake detectors.

---

# Attribution

This repository contains an **independent research and educational implementation**.

The implementation is inspired by:

1. The hybrid deepfake detection research described in the referenced ACM paper.
2. The DeepfakeBench framework and its standardized deepfake detection workflow.
3. The Self-Blended Images methodology used for Stage 1 data generation.

The code in this repository should **not** be interpreted as the official implementation of the referenced paper or DeepfakeBench.

Third-party datasets, pretrained models, checkpoints, and external source code remain subject to their respective licenses and terms.

---

# Limitations

This project is an experimental research implementation and has several limitations.

- The hybrid model is primarily evaluated on face-frame based data.
- The current main configuration uses FaceForensics++ c23.
- Generalization to completely unseen datasets and manipulation methods requires further validation.
- Large pretrained visual encoders make training computationally expensive.
- Video-level prediction is obtained by aggregating frame-level scores rather than modeling temporal information directly.
- The model should not be treated as a forensic-grade production system without additional validation.

---

# Author

**TrThuc**

GitHub:

https://github.com/TrThuc/Hybird_Model_for_DF

---

# Disclaimer

This project is intended for **research and educational purposes**.

Deepfake detection results should be interpreted as model predictions rather than definitive forensic evidence.
