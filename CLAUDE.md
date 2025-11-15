# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Completed Computer Vision Project**: Leukocyte (white blood cell) classification using ResNet18 with transfer learning.

**Current Status**: ✅ Production-ready model achieving 99.47% test accuracy and 100% external validation.

**Dataset**: 2,500 labeled images (500 per class) of 5 leukocyte types: basophil, eosinophil, lymphocyte, monocyte, neutrophil.

**Key Achievement**: ResNet18 model achieves:
- Test Set: 99.47% accuracy (373/375 correct)
- External Dataset: 100% accuracy (9/9 monocyte images)
- Fully reproducible with seed=42

## Quick Start Commands

### Environment Setup
```bash
# Activate conda environment (auto-activated via direnv)
conda activate cv

# Verify installation
python -c "import torch, fastai; print(f'PyTorch: {torch.__version__}, fastai: {fastai.__version__}')"
```

### Running the Pipeline

```bash
# 1. Data Preparation (creates train/val/test split)
python 01_data_preparation.py

# 2. Model Training (run notebook)
jupyter notebook notebooks/02_model_training.ipynb

# 3. Model Evaluation (run notebook)
jupyter notebook notebooks/03_model_evaluation.ipynb
```

### Model Testing

```bash
# Verify model loads correctly
python -c "from fastai.vision.all import load_learner; learn = load_learner('outputs/model.pkl'); print('✓ Model loads successfully')"
```

### Report Generation

```bash
# Compile LaTeX PDF report (2 pages)
pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
rm -f report.aux report.log report.out

# Or use the helper script
./compile_report.sh
```

## Project Architecture

### Three-Stage Pipeline

**Stage 1: Data Preparation** (`01_data_preparation.py`)
- Loads 2,500 images from `Dataset and Notebook-20251115/dataset_leukocytes/`
- Creates stratified 70/15/15 train/val/test split
- Generates visualization figures
- **Output**: `outputs/data_split.csv` (reproducible split with seed=42)

**Stage 2: Model Training** (`notebooks/02_model_training.ipynb`)
- Architecture: ResNet18 (pretrained on ImageNet)
- Two-phase training:
  - Phase 1: 20 epochs frozen backbone (LR=0.001, patience=3)
  - Phase 2: 20 epochs fine-tuning (LR=0.0001, patience=5)
- Data augmentation: rotations (±180°), flips, crops (75-100%), warping, lighting
- Early stopping to prevent overfitting
- **Output**: `outputs/model.pkl` (84MB), `outputs/model_metadata.json`

**Stage 3: Model Evaluation** (`notebooks/03_model_evaluation.ipynb`)
- Test set evaluation (375 images)
- External dataset validation (9 monocyte images)
- Confusion matrices and per-class metrics
- **Output**: 13 visualization figures in `outputs/figures/`

### Key Design Decisions

**Why ResNet18 (not ResNet34/50)?**
- 11.7M parameters vs 21.8M for ResNet34 (46% reduction)
- Achieves perfect external validation (100%)
- No overfitting on 2,500 image dataset
- Faster training and inference
- Optimal capacity for 5-class task

**Two-Phase Training Strategy**:
1. Freeze backbone, train classifier head → adapt to leukocyte domain
2. Unfreeze all layers, fine-tune → optimize deep features
3. Early stopping prevents overfitting

**Data Augmentation**:
- Geometric: rotation ±180°, flips, crop 75-100%, warp factor 0.2
- Photometric: brightness/contrast factor 0.5
- Application probability: 75% for affine and lighting transforms

### Critical Constraints

**Model Export/Loading**:
- ✅ **VERIFIED**: Model loads with `load_learner('model.pkl')` only
- ❌ No custom functions, libraries, or modifications allowed
- Export includes full model state, data preprocessing, and class vocabulary

**Reproducibility**:
- All random operations seeded with `seed=42`
- Use `utils.set_seed(42)` to set Python, NumPy, PyTorch (CPU/CUDA/MPS) seeds
- Data split saved to `outputs/data_split.csv` for consistency
- Model metadata tracked in `outputs/model_metadata.json`

## File Structure

```
cv-nhan/
├── 01_data_preparation.py          # Data splitting script
├── utils.py                        # Seed management utilities
├── notebooks/
│   ├── 02_model_training.ipynb     # Two-phase ResNet18 training
│   └── 03_model_evaluation.ipynb   # Comprehensive evaluation
├── outputs/
│   ├── data_split.csv              # 70/15/15 split (reproducible)
│   ├── model.pkl                   # Trained ResNet18 (99.47% test, 100% external)
│   ├── model_metadata.json         # Export timestamp, hyperparams, metrics
│   └── figures/                    # 13 visualization figures
├── report.tex                      # LaTeX source for 2-page PDF
├── report.pdf                      # Scientific paper format
├── PROJECT_REPORT.md               # Comprehensive markdown report (16KB)
├── REPRODUCIBILITY.md              # Seed management documentation
└── requirements.txt                # Python dependencies
```

## Development Environment

**Conda Environment**: `cv` (Python 3.11.8)
- Auto-activated via direnv (`.envrc`)
- MPS acceleration for Apple Silicon

**Key Dependencies**:
- PyTorch 2.9.1
- fastai 2.8.5
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Pillow

## Model Performance Summary

| Metric | Test Set | External Dataset |
|--------|----------|------------------|
| Accuracy | 99.47% | 100% |
| Errors | 2/375 | 0/9 |
| Per-class | ~100% all metrics | Perfect monocyte detection |

**Only Errors**: 2 basophils misclassified as neutrophils (morphologically similar granulocytes)

## Deliverables

**Submission Package**:
```
LastName1_LastName2_LastName3_LastName4.zip
├── model.pkl          # 84MB, ResNet18
└── report.pdf         # 2 pages, 622KB
```

**Report Contents**:
- Architecture: ResNet18 with two-phase training
- Results: 99.47% test, 100% external
- Data augmentation details
- Confusion matrices and training curves
- Error analysis

## Common Tasks

### Re-run Training
```bash
# Start fresh training (will save to outputs/model.pkl)
jupyter notebook notebooks/02_model_training.ipynb
# Run all cells, final cell exports model with verification
```

### Re-generate Figures
```bash
# Run evaluation notebook to regenerate all 13 figures
jupyter notebook notebooks/03_model_evaluation.ipynb
```

### Update Reports
```bash
# Regenerate PDF report
pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
rm -f report.aux report.log report.out

# PROJECT_REPORT.md is manually maintained markdown
```

### Verify Reproducibility
```bash
# Re-run data preparation (should produce identical data_split.csv)
python 01_data_preparation.py

# Check CSV matches
md5 outputs/data_split.csv  # Compare with previous hash
```

## Important Notes

**Seed Management**:
- Always call `set_seed(42)` at the start of notebooks/scripts
- Uses `utils.set_seed()` to set all library seeds (Python, NumPy, PyTorch)
- fastai's `set_seed(42, reproducible=True)` for additional determinism

**Model Export Verification**:
- Training notebook (cell 34) includes 5-step verification:
  1. Load best model checkpoint
  2. Verify validation performance
  3. Test on external dataset
  4. Export model
  5. Verify exported model loads

**External Dataset Limitation**:
- Contains only monocyte images (9 samples)
- 100% accuracy achieved, but only tests 1 of 5 classes
- Full multi-class external validation recommended for production

**LaTeX Compilation**:
- Requires pdflatex (install via MacTeX or BasicTeX)
- Compile twice for cross-references
- Clean up auxiliary files (.aux, .log, .out)

## References

- [README.md](README.md) - Quick start and overview
- [PROJECT_REPORT.md](PROJECT_REPORT.md) - Comprehensive analysis with figures
- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) - Detailed architecture decisions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Seed management guide
- [report.pdf](report.pdf) - 2-page scientific paper
