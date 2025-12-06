# Leukocyte Classification Project

Deep learning model for classifying five types of white blood cells (leukocytes) using computer vision.

## Project Overview

- **Course**: Computer Vision (Curs 295II022)
- **Objective**: Develop a CNN/ViT model to classify leukocytes with >=92% accuracy on external dataset
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
# 1. 01_data_preparation.py (or notebook)
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
- Train CNN (ResNet34) with two-phase approach
- Strong color augmentation for stain robustness
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
│   ├── 02_model_training.ipynb      # Model training (ResNet34, two-phase)
│   └── 03_model_evaluation.ipynb    # Comprehensive evaluation
├── outputs/
│   ├── data_split.csv               # Train/val/test split (70/15/15)
│   ├── model.pkl                    # Trained model (99.20% test, 100% external)
│   ├── model_metadata.json          # Model export metadata
│   └── figures/                     # Visualization figures
├── 01_data_preparation.py           # Data splitting script
├── utils.py                         # Seed management utilities
├── stain_normalization.py           # Stain normalization utilities (optional)
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
- GPU acceleration via CUDA/MPS

### Model Export/Loading
**VERIFIED**: Model loads correctly using only:
```python
from fastai.vision.all import load_learner
learn = load_learner('outputs/model.pkl')
```

- No custom functions required
- No additional libraries needed
- Direct loading works perfectly

### Grading Criteria
- External dataset accuracy >= 92% = Grade 10
- **Our achievement: 100% external accuracy**
- Model loads correctly with `load_learner()`

## Current Implementation

### Architecture: ResNet34
- **Model:** ResNet34 (pretrained on ImageNet)
- **Image size:** 224x224
- **Batch size:** 32
- **Training:** Two-phase approach
  - Phase 1: 30 epochs (frozen backbone, LR=0.001, patience=8)
  - Phase 2: 30 epochs (fine-tuned, LR=0.0001, patience=8)
- **Data Augmentation:** Strong color augmentation for stain robustness
  - Brightness/Contrast: +/-40%
  - Saturation: +/-40%
  - Hue: +/-10%
  - Rotation: +/-180 degrees
  - Random crop: 75-100%
- **Achieved accuracy:**
  - Validation: 99.47%
  - Test set: 99.20%
  - External validation: 100%

### Why ResNet34 with Color Augmentation?
- Excellent performance (99.20% test, 100% external)
- Strong color augmentation ensures robustness to staining variations
- Two-phase training with early stopping prevents overfitting
- Perfect for this dataset size (2,500 images)

## Important Notes

1. **Reproducibility**: Use saved `data_split.csv` for all experiments
2. **Stain Robustness**: Color augmentation (saturation, hue, brightness) handles different staining protocols
3. **External Testing**: Regularly test on `test_second_dataset/`
4. **Model Export**: Always verify model loads before submission

## Deliverables

1. **PDF Report** (2 pages): `report.pdf`
   - Architecture: ResNet34
   - Training strategy: Two-phase with early stopping
   - Results: 99.20% test, 100% external
   - Data augmentation details
   - Confusion matrices and metrics

2. **Model File**: `outputs/model.pkl`
   - Architecture: ResNet34
   - Loads with: `load_learner('model.pkl')`
   - Performance: 99.20% test, 100% external

3. **Submission Package**:
   ```
   LastName1_LastName2_LastName3_LastName4.zip
   ├── model.pkl          # ResNet34
   └── report.pdf         # 2 pages
   ```

## Results

**[VIEW COMPREHENSIVE PROJECT REPORT](PROJECT_REPORT.md)** - Complete analysis with all figures and metrics
**[VIEW 2-PAGE PDF REPORT](report.pdf)** - Professional scientific paper format

**Quick Summary:**
- Architecture: **ResNet34** with strong color augmentation
- Validation Accuracy: **99.47%** (373/375 correct)
- Test Accuracy: **99.20%** (372/375 correct)
- External Validation: **100%** (9/9 perfect predictions on monocyte dataset)
- All classes: ~100% precision/recall/F1-score
- Fully reproducible with seed=42

## Documentation

### Reports
- [report.pdf](report.pdf) - **2-page scientific paper** (submission format)
- [PROJECT_REPORT.md](PROJECT_REPORT.md) - **Comprehensive analysis** (all figures)

### Technical Documentation
- [CLAUDE.md](CLAUDE.md) - Claude Code quick reference
- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) - Architecture and design decisions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Reproducibility guide (seed=42)

### External Resources
- [fastai documentation](https://docs.fast.ai/)
- [Project PDF](Project_aplication_Computer_vision.pdf) - Official course guidelines

## Submission Checklist

### Required Actions Before Submission

**1. Add Student Information to Report**
The PDF report needs student names and IDs. Either:
- Edit `report.tex` and recompile, OR
- Use a PDF editor to add names/IDs to `report.pdf`

**2. Test Model with Provided Notebook**
Verify model loads correctly using the exact test procedure:
```bash
# Copy model to test location
cp outputs/model.pkl "Dataset and Notebook-20251115/"

# Open and run the provided test notebook
jupyter notebook "Dataset and Notebook-20251115/test your model.ipynb"
```

**3. Create Submission ZIP**
```bash
# Prepare submission folder
mkdir -p submission
cp outputs/model.pkl submission/
cp report.pdf submission/

# Create ZIP (replace with actual student last names)
cd submission
zip ../Garcia_Lopez_Martinez_Rodriguez.zip model.pkl report.pdf
```

### Final Verification Checklist

- [ ] Student names and IDs added to report.pdf
- [ ] Model tested with `test your model.ipynb`
- [ ] Model loads with `load_learner()` only
- [ ] ZIP file named correctly (LastName_LastName_LastName_LastName.zip)
- [ ] ZIP contains both model.pkl and report.pdf

### Performance Summary

**External Dataset:**
- Achieved: **100% accuracy** (9/9 monocyte images)
- Requirement: >=92% for grade 10

**Requirements Met:**
- Model loads with `load_learner()` only
- 2-page PDF report with all sections
- Training/validation/test plots
- Confusion matrix
- Complete reproducibility

## Deadline

**Due:** December 11, 2025, 23:59 CEST

**Important:**
- Late submissions: **30% deduction per day**
- Plagiarism: **Zero tolerance** - all involved students receive grade 0
- Submit to: **Atenea** (Block 2)

---

**Good luck with your submission!**
