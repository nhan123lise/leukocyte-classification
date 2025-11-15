# Leukocyte Classification Project

Deep learning model for classifying five types of white blood cells (leukocytes) using computer vision.

## Project Overview

- **Course**: Computer Vision (Curs 295II022)
- **Objective**: Develop a CNN/ViT model to classify leukocytes with ≥92% accuracy on external dataset
- **Dataset**: 2,500 labeled images (500 per class)
- **Classes**: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil

## Quick Start

### 1. Environment Setup

```bash
# The cv conda environment will be auto-activated via direnv
cd /Users/mac/code/cv-nhan

# Verify installation
python -c "import torch, fastai; print(f'PyTorch: {torch.__version__}, fastai: {fastai.__version__}')"
```

### 2. Project Workflow

The project is divided into three stages, each with its own Jupyter notebook:

```bash
# Start Jupyter
jupyter notebook

# Then run notebooks in order:
# 1. notebooks/01_data_preparation.ipynb
# 2. notebooks/02_model_training.ipynb
# 3. notebooks/03_model_evaluation.ipynb
```

### Stage 1: Data Preparation
- Load and explore dataset
- Create train/validation/test split (70/15/15)
- Save split configuration
- **Output**: `outputs/data_split.csv`

### Stage 2: Model Training
- Load data using fastai
- Train CNN (ResNet34 recommended) or ViT
- Optimize hyperparameters
- Export trained model
- **Output**: `outputs/model.pkl`

### Stage 3: Model Evaluation
- Evaluate on test set
- Generate confusion matrix and metrics
- Test on external dataset
- Analyze errors
- **Output**: Figures for final report

## Project Structure

```
cv-nhan/
├── Dataset and Notebook-20251115/
│   ├── dataset_leukocytes/          # Main training dataset
│   │   ├── basophil/                # 500 images
│   │   ├── eosinophil/              # 500 images
│   │   ├── lymphocyte/              # 500 images
│   │   ├── monocyte/                # 500 images
│   │   └── neutrophil/              # 500 images
│   ├── test_second_dataset/         # External validation (monocyte only)
│   └── test your model.ipynb        # Model loading template
├── notebooks/
│   ├── 02_model_training.ipynb      # Model training (ResNet18, two-phase)
│   └── 03_model_evaluation.ipynb    # Comprehensive evaluation
├── outputs/
│   ├── data_split.csv               # Train/val/test split (70/15/15)
│   ├── model.pkl                    # Trained model (99.47% test, 100% external)
│   ├── model_metadata.json          # Model export metadata
│   └── figures/                     # 13 visualization figures
├── 01_data_preparation.py           # Data splitting script
├── utils.py                         # Seed management utilities
├── report.tex                       # LaTeX source for PDF report
├── report.pdf                       # 2-page scientific report
├── PROJECT_REPORT.md                # Comprehensive markdown report
├── .envrc                           # direnv configuration (conda cv)
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── CLAUDE.md                        # Claude Code documentation
├── SYSTEM_DESIGN.md                 # Detailed architecture guide
├── REPRODUCIBILITY.md               # Reproducibility documentation
└── README.md                        # This file
```

## Key Requirements

### Technical
- Python 3.11+
- PyTorch 2.9.1
- fastai 2.8.5
- GPU acceleration via MPS (Apple Silicon)

### Model Export/Loading ✅
**VERIFIED**: Model loads correctly using only:
```python
from fastai.vision.all import load_learner
learn = load_learner('outputs/model.pkl')
```

✅ No custom functions required
✅ No additional libraries needed
✅ Direct loading works perfectly

### Grading Criteria
- External dataset accuracy ≥ 92% = Grade 10
- **Our achievement: 100% external accuracy** 🎯
- Model loads correctly with `load_learner()` ✅

## Current Implementation

### Architecture: ResNet18 ✅
- **Model:** ResNet18 (pretrained on ImageNet)
- **Image size:** 224x224
- **Batch size:** 32
- **Training:** Two-phase approach
  - Phase 1: 20 epochs (frozen backbone, LR=0.001)
  - Phase 2: 20 epochs (fine-tuned, LR=0.0001)
- **Early Stopping:** Patience=3 (phase 1), Patience=5 (phase 2)
- **Data Augmentation:** Extensive geometric and photometric transforms
- **Achieved accuracy:**
  - Test set: 99.47%
  - External validation: 100%

### Why ResNet18?
- ✅ Excellent performance (99.47% test, 100% external)
- ✅ Lightweight and efficient (fewer parameters than ResNet34/50)
- ✅ Faster training and inference
- ✅ Perfect for this dataset size (2,500 images)

## Important Notes

1. **Reproducibility**: Use saved `data_split.csv` for all experiments
2. **Overfitting**: Monitor train-validation gap; use augmentation
3. **External Testing**: Regularly test on `test_second_dataset/`
4. **Model Export**: Always verify model loads before submission

## Deliverables ✅

1. **PDF Report** (2 pages): ✅ `report.pdf`
   - Architecture: ResNet18
   - Training strategy: Two-phase with early stopping
   - Results: 99.47% test, 100% external
   - Data augmentation details
   - Confusion matrices and metrics
   - Complete methodology

2. **Model File**: ✅ `outputs/model.pkl`
   - Size: 84MB
   - Architecture: ResNet18
   - Loads with: `load_learner('model.pkl')`
   - Performance: 99.47% test, 100% external

3. **Submission Package**:
   ```
   LastName1_LastName2_LastName3_LastName4.zip
   ├── model.pkl          # 84MB, ResNet18
   └── report.pdf         # 2 pages, 622KB
   ```

## Results

📊 **[VIEW COMPREHENSIVE PROJECT REPORT](PROJECT_REPORT.md)** - Complete analysis with all figures and metrics
📄 **[VIEW 2-PAGE PDF REPORT](report.pdf)** - Professional scientific paper format

**Quick Summary:**
- ✅ Architecture: **ResNet18** (efficient and lightweight)
- ✅ Test Accuracy: **99.47%** (373/375 correct)
- ✅ External Validation: **100%** (9/9 perfect predictions on monocyte dataset)
- ✅ All classes: ~100% precision/recall/F1-score
- ✅ Fully reproducible with seed=42

## Documentation

### Reports
- [report.pdf](report.pdf) - **2-page scientific paper** (submission format)
- [PROJECT_REPORT.md](PROJECT_REPORT.md) - **Comprehensive analysis** (16KB, all figures)

### Technical Documentation
- [CLAUDE.md](CLAUDE.md) - Claude Code quick reference
- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) - Architecture and design decisions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Reproducibility guide (seed=42)

### External Resources
- [fastai documentation](https://docs.fast.ai/)
- [Project PDF](Project_aplication_Computer_vision.pdf) - Official course guidelines

## Deadline

December 11, 2025, 23:59 CEST

---

**Good luck!** 🚀
