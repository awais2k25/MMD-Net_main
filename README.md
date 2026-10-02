# MMD-Net: Multi-Modal Deepfake Detection Network

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MMD-Net** is an advanced multimodal deepfake detection framework designed to detect manipulated audio-visual media. By leveraging dual-branch feature disentanglement (forgery-specific vs. identity/content-irrelevant features), cross-modal attention fusion, and multi-task learning objectives, MMD-Net achieves high generalization and robust performance across intra-dataset and cross-dataset benchmarks (including **FakeAVCeleb**, **DFDC**, and **KoDF**).

---

## 🌟 Key Features

- **Multimodal Dual-Branch Disentanglement**:
  - **Visual Encoder**: EfficientNet backbone (`b2` default) extracting visual forgery-specific (`vfs`) and visual identity/irrelevant (`vir`) representations.
  - **Audio Encoder**: Hybrid Frequency + Temporal Convolutional Network (TCN) extracting audio forgery-specific (`afs`) and audio identity/irrelevant (`air`) representations.
  - **Orthogonality Regularization**: Enforces strict independence between forgery-specific and identity-related features within each modality.
- **Cross-Modal Attention Fusion (MMT / CMAF)**:
  - Multimodal Transformer and Cross-Modal Attention Fusion aligning audio and visual temporal representations.
- **Uncertainty-Weighted Multi-Loss Optimization**:
  - Dynamic loss balancing across binary classification, multi-class classification, orthogonality loss, triplet loss, and optional reconstruction loss.
- **Dual Classification Output**:
  - **Binary Classification**: Real vs. Fake prediction.
  - **Multi-Class Classification**: Fine-grained categorization (Real, Fake Video + Real Audio, Real Video + Fake Audio, Fake Video + Fake Audio).
- **Comprehensive Evaluation & Analytics**:
  - ROC-AUC, PR-AUC, Confusion Matrix generation, Balanced Accuracy, TPR/FPR/TNR/FNR analysis.
  - Test-Time Augmentation (TTA) support for robust testing.
  - Built-in t-SNE feature space visualization and reconstruction comparison tooling.

---

## 📁 Repository Structure

```text
MMD-Net_main/
├── Main_v3.py                   # Main training, validation, testing & evaluation pipeline
├── check_efficientnet_models.py # Model verification & test script for EfficientNet backbones
├── check_mels_corpt_shpaes.py   # Audio mel-spectrogram validation utility
├── frame_shapes_corpt_check.py  # Video frame dimension & shape integrity checker
├── generate_charts.py           # Training metrics and loss visualization generator
├── requirements.txt             # Python dependencies
├── trace_shapes.py              # Tensor shape tracing & pipeline diagnostics
├── data/
│   ├── MMD_dataset.py           # Dataset loader
│   ├── MMD_dataset_v4.py        # Optimized multimodal dataset loader (static + temporal)
│   ├── consts.py                # Dataset configuration constants
│   └── actual_dataset_shapes.py # Dataset dimension inspection
├── models/
│   ├── MMD_NET.py               # Core MMD-Net model wrapper
│   ├── MMT.py                   # Multimodal Transformer & CMAF fusion module
│   ├── efficient_net.py         # Dual-branch EfficientNet visual encoder
│   ├── AudioEncoderTCN.py       # Dual-branch Freq+TCN audio encoder
│   ├── classifier_heads.py      # Binary & Multi-class classification heads
│   ├── Reconstruction.py        # Visual & Audio decoder modules (for ablation studies)
│   └── audio_encoder/           # Audio encoder components (TCN, Frequency branch, etc.)
└── utils/
    ├── Losses_v2.py / Losses_v3.py # Uncertainty-weighted loss formulations
    ├── TSNE_Feature_Viz.py      # t-SNE feature embedding visualization
    ├── Recon_Comp_Viz.py        # Reconstruction comparison generator
    └── debug_utils.py           # Shape checking and debugging utilities
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/awais2k25/MMD-Net_main.git
cd MMD-Net_main
```

### 2. Set Up Python Environment
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

MMD-Net is structured to support multimodal deepfake datasets such as:
- **FakeAVCeleb** (balanced splits)
- **DFDC** (Deepfake Detection Challenge)
- **KoDF** (Korean DeepFake)

### Preprocessing Expectations:
1. **Video Frames**: Cropped & aligned face frames (default 25 frames per clip, resized to standard spatial resolution e.g., $100 \times 100$ or $224 \times 224$).
2. **Audio Mel-Spectrograms**: Converted to single/mono audio channel with fixed time-frequency representation ($1 \times 128 \times 32$ per window).

Configure the dataset directory in `Main_v3.py` or through your custom config script.

---

## ⚡ Usage

### Training MMD-Net
To train the model from scratch or resume from a checkpoint:
```bash
python Main_v3.py
```
*(Configure `test_mode = False` inside `Main_v3.py` `Config` class).*

Key training configurations:
- **Backbones**: Visual (`b2`, `b0`, etc.) and Audio (`18`, etc.)
- **Batch Size & Accumulation**: Default batch size of 20 with gradient accumulation steps.
- **Optimizer**: AdamW with weight decay and Cosine Annealing learning rate schedule.

### Testing & Evaluation
To evaluate a trained checkpoint on test or cross-dataset splits:
```python
# In Main_v3.py Config class:
test_mode = True
use_tta = True  # Enable Test-Time Augmentation
trained_model_path = "path/to/checkpoint.pth"
```
Run testing:
```bash
python Main_v3.py
```

### Visualizations & Diagnostics

#### 1. Generate Training Metric Charts
```bash
python generate_charts.py
```

#### 2. t-SNE Feature Space Visualization
Enable `visualize_tsne = True` in `Main_v3.py` or run directly using `utils/TSNE_Feature_Viz.py` to inspect feature separation between genuine and manipulated samples.

#### 3. Shape and Integrity Checks
```bash
python trace_shapes.py
python frame_shapes_corpt_check.py
python check_mels_corpt_shpaes.py
```

---

## 📈 Metric Outputs

The evaluation pipeline produces detailed classification metrics:
- **Binary Metrics**: Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1, Balanced Accuracy, TPR, FPR, TNR, FNR.
- **Multi-Class Metrics**: Per-class precision, recall, F1, confusion matrices, and macro-averaged ROC/PR AUC.
- **Automated Plotting**: Saved high-resolution `.png` plots for ROC curves, PR curves, and Confusion Matrices.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
