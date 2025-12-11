# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Completed Computer Vision Project**: Leukocyte (white blood cell) classification using ResNet34 with transfer learning.

**Current Status**: Production-ready model achieving 98.93% test accuracy, 100% validation accuracy, and 100% external validation.

**Dataset**: 2,500 labeled images (500 per class) of 5 leukocyte types: basophil, eosinophil, lymphocyte, monocyte, neutrophil.

**Key Achievement**: ResNet34 model achieves:
- Validation Set: 100% accuracy (375/375 correct)
- Test Set: 98.93% accuracy (371/375 correct)
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

# 2. External Data Preparation (deduplicates and organizes external datasets)
python 02_external_data_preparation.py

# 3. Model Training (run notebook)
jupyter notebook notebooks/02_model_training.ipynb

# 4. Model Evaluation (run notebook)
jupyter notebook notebooks/03_model_evaluation.ipynb
```

### Model Testing

```bash
# Verify model loads correctly
python -c "from fastai.vision.all import load_learner; learn = load_learner('outputs/model.pkl'); print('Model loads successfully')"
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
- Architecture: ResNet34 (pretrained on ImageNet)
- Two-phase training:
  - Phase 1: 30 epochs frozen backbone (LR=0.001, patience=8)
  - Phase 2: 30 epochs fine-tuning (LR=0.00001, patience=8)
- Data augmentation for stain robustness:
  - Geometric: rotation ±180°, flips, crop 75-100%, warp 0.2
  - Color: brightness/contrast ±40%, saturation ±40%, hue ±10%
- Early stopping to prevent overfitting
- **Output**: `outputs/model.pkl`, `outputs/model_metadata.json`

**Stage 3: Model Evaluation** (`notebooks/03_model_evaluation.ipynb`)
- Test set evaluation (375 images)
- External dataset validation (9 monocyte images)
- Confusion matrices and per-class metrics
- **Output**: Visualization figures in `outputs/figures/`

**External Data Preparation** (`02_external_data_preparation.py`)
- Loads external datasets from `data_external/`:
  - `dataset/`: 16,633 images (Train/Test-A/Test-B)
  - `PBC_dataset_normal_DIB/`: 10,299 target class images
- Normalizes class names (handles case differences)
- Filters to 5 target classes (excludes erythroblast, ig, platelet)
- Deduplicates using MD5 (exact) and pHash/dHash (perceptual)
- Removes duplicates of original training data (prevents data leakage)
- **Output**: `data_external_unified/` (24,417 unique images), `outputs/external_manifest.csv`

**Deduplication Module** (`deduplication.py`)
- Reusable image deduplication with MD5, pHash, dHash algorithms
- Priority-based resolution (lower priority kept on duplicates)
- Cross-dataset duplicate detection
- Configurable similarity thresholds (default: Hamming distance <= 8)

### Key Design Decisions

**Why ResNet34?**
- 21.8M parameters - good capacity for 5-class medical imaging task
- Achieves 98.93% test accuracy, 100% validation accuracy, and 100% external validation
- Strong color augmentation ensures robustness to staining variations
- Two-phase training with early stopping prevents overfitting

**Two-Phase Training Strategy**:
1. Freeze backbone, train classifier head -> adapt to leukocyte domain
2. Unfreeze all layers, fine-tune -> optimize deep features
3. Early stopping (patience=8) prevents overfitting

**Data Augmentation for Stain Robustness**:
- Geometric: rotation ±180°, flips, crop 75-100%, warp factor 0.2
- Color (via fastai):
  - `max_lighting=0.4` (brightness/contrast ±40%)
  - `Saturation(max_lighting=0.4, p=0.75)` (saturation ±40%)
  - `Hue(max_hue=0.1, p=0.75)` (hue shift ±10%)
- High probability: `p_affine=0.75`, `p_lighting=0.9`

### Critical Constraints

**Model Export/Loading**:
- **VERIFIED**: Model loads with `load_learner('model.pkl')` only
- No custom functions, libraries, or modifications required
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
├── 02_external_data_preparation.py # External data dedup & organization
├── deduplication.py                # Reusable image deduplication module
├── stain_normalization.py          # Stain normalization utilities (optional)
├── utils.py                        # Seed management utilities
├── notebooks/
│   ├── 02_model_training.ipynb     # Two-phase ResNet34 training
│   └── 03_model_evaluation.ipynb   # Comprehensive evaluation
├── data_external/                  # Raw external datasets
│   ├── dataset/                    # 16,633 images (Train/Test-A/Test-B)
│   └── PBC_dataset_normal_DIB/     # 17,093 images (8 classes)
├── data_external_unified/          # Deduplicated external validation set
│   ├── basophil/                   # 1,013 images
│   ├── eosinophil/                 # 3,677 images
│   ├── lymphocyte/                 # 4,323 images
│   ├── monocyte/                   # 1,714 images
│   └── neutrophil/                 # 13,690 images
├── outputs/
│   ├── data_split.csv              # 70/15/15 split (reproducible)
│   ├── model.pkl                   # Trained ResNet34 (98.93% test, 100% val, 100% external)
│   ├── model_metadata.json         # Export timestamp, hyperparams, metrics
│   ├── external_manifest.csv       # External dataset manifest with hashes
│   ├── dedup_report.json           # Deduplication statistics
│   └── figures/                    # Visualization figures
├── report.tex                      # LaTeX source for 2-page PDF
├── report.pdf                      # Scientific paper format
├── PROJECT_REPORT.md               # Comprehensive markdown report
├── REPRODUCIBILITY.md              # Seed management documentation
└── requirements.txt                # Python dependencies
```

## Development Environment

**Conda Environment**: `cv` (Python 3.11)
- Auto-activated via direnv (`.envrc`)
- CUDA/MPS acceleration supported

**Key Dependencies**:
- PyTorch 2.9.1
- fastai 2.8.5
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Pillow
- imagehash (for deduplication)
- torchstain (for stain normalization)

## Model Performance Summary

| Metric | Validation | Test Set | External Dataset |
|--------|------------|----------|------------------|
| Accuracy | 100% | 98.93% | 100% |
| Errors | 0/375 | 4/375 | 0/9 |

**Test Set Errors** (4 total):
- 2 monocytes misclassified
- 1 lymphocyte misclassified
- 1 neutrophil misclassified

**Per-Class Test Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| basophil | 1.0000 | 1.0000 | 1.0000 |
| eosinophil | 1.0000 | 1.0000 | 1.0000 |
| lymphocyte | 0.9867 | 0.9867 | 0.9867 |
| monocyte | 0.9865 | 0.9733 | 0.9799 |
| neutrophil | 0.9737 | 0.9867 | 0.9801 |

## Deliverables

**Submission Package**:
```
LastName1_LastName2_LastName3_LastName4.zip
├── model.pkl          # ResNet34
└── report.pdf         # 2 pages
```

**Report Contents**:
- Architecture: ResNet34 with two-phase training
- Results: 98.93% test, 100% validation, 100% external
- Data augmentation details (color augmentation for stain robustness)
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
# Run evaluation notebook to regenerate all figures
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

### Reprocess External Data
```bash
# Re-run external data preparation (regenerates unified dataset)
python 02_external_data_preparation.py

# Check deduplication results
cat outputs/dedup_report.json | python -m json.tool
```

## Important Notes

**Seed Management**:
- Always call `set_seed(42)` at the start of notebooks/scripts
- Uses `utils.set_seed()` to set all library seeds (Python, NumPy, PyTorch)
- fastai's `set_seed(42, reproducible=True)` for additional determinism

**Model Export Verification**:
- Training notebook includes 5-step verification:
  1. Load best model checkpoint
  2. Verify validation performance
  3. Test on external dataset
  4. Export model
  5. Verify exported model loads

**External Validation Dataset**:
- Original: 9 monocyte samples (100% accuracy achieved)
- Expanded: 24,417 deduplicated images across all 5 classes from `data_external_unified/`
  - basophil: 1,013 | eosinophil: 3,677 | lymphocyte: 4,323 | monocyte: 1,714 | neutrophil: 13,690
- 2,514 exact duplicates removed (MD5 match with original training data)
- Uses symlinks to save disk space

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
