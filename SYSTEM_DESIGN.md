# Leukocyte Classification System Design

## Project Goal
Develop a deep learning model to classify 5 types of white blood cells (leukocytes) with ≥92% accuracy on external dataset.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LEUKOCYTE CLASSIFICATION SYSTEM              │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          ┌─────────▼─────────┐         ┌────────▼────────┐
          │   DATA PIPELINE   │         │  MODEL PIPELINE │
          └─────────┬─────────┘         └────────┬────────┘
                    │                            │
        ┌───────────┼───────────┐                │
        │           │           │                │
   ┌────▼────┐ ┌───▼────┐ ┌───▼────┐      ┌────▼─────┐
   │  Train  │ │  Val   │ │  Test  │      │  Model   │
   │ 60-70%  │ │ 15-20% │ │ 15-20% │      │ (PKL)    │
   └─────────┘ └────────┘ └────────┘      └──────────┘
```

## Three-Stage Development Pipeline

### Stage 1: Data Preparation & Exploration
**Notebook**: `01_data_preparation.ipynb`

**Objectives**:
- Load and explore dataset structure
- Verify class distribution (500 images per class)
- Perform exploratory data analysis (EDA)
- Create train/validation/test split
- Save split configuration for reproducibility

**Key Tasks**:
1. **Data Loading**
   - Path: `Dataset and Notebook-20251115/dataset_leukocytes/`
   - Classes: basophil, eosinophil, lymphocyte, monocyte, neutrophil
   - Total: 2,500 images (500 per class)

2. **Exploratory Data Analysis**
   - Image dimensions analysis
   - Visualize sample images from each class
   - Check for data quality issues
   - Analyze pixel intensity distributions

3. **Data Splitting Strategy**
   - Train: 60-70% (1,500-1,750 images)
   - Validation: 15-20% (375-500 images)
   - Test: 15-20% (375-500 images)
   - Stratified split to maintain class balance

4. **Output**
   - `data_split.csv` or `data_split.pkl` with file paths and split labels
   - EDA visualizations and statistics

**Code Structure**:
```python
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load dataset
# 2. Create dataframe with paths and labels
# 3. EDA and visualizations
# 4. Stratified split
# 5. Save split information
```

---

### Stage 2: Model Training & Optimization
**Notebook**: `02_model_training.ipynb`

**Objectives**:
- Design CNN or Vision Transformer architecture
- Train model using fastai
- Optimize hyperparameters
- Monitor training progress
- Export best model

**Architecture Options**:

#### Option A: Transfer Learning with CNN (Recommended)
```python
from fastai.vision.all import *

# ResNet50/ResNet34 with transfer learning
learn = vision_learner(
    dls,
    resnet50,  # or resnet34, resnet18
    metrics=[accuracy, error_rate]
)
```

**Advantages**:
- Proven performance on medical images
- Faster training
- Less data required
- Good baseline performance

#### Option B: Vision Transformer (ViT)
```python
from fastai.vision.all import *

# Vision Transformer
learn = vision_learner(
    dls,
    vit_base_patch16_224,  # or vit_small_patch16_224
    metrics=[accuracy, error_rate]
)
```

**Advantages**:
- State-of-the-art potential
- Better global context understanding
- May achieve higher accuracy

**Training Strategy**:

1. **Data Augmentation** (Critical for generalization)
   ```python
   # Recommended augmentations for medical images
   aug_transforms(
       size=224,
       min_scale=0.75,
       max_rotate=15.0,
       max_lighting=0.2,
       max_warp=0.2,
       p_affine=0.75,
       p_lighting=0.75
   )
   ```

2. **Learning Rate Finding**
   ```python
   learn.lr_find()  # Find optimal learning rate
   ```

3. **Training Schedule**
   - Phase 1: Freeze backbone, train head (3-5 epochs)
   - Phase 2: Unfreeze, fine-tune with discriminative learning rates (10-20 epochs)

4. **Regularization Techniques**
   - Dropout (built into architectures)
   - Weight decay
   - Mixup augmentation (optional)
   - Early stopping based on validation loss

**Hyperparameters to Experiment With**:
- Learning rate: 1e-4 to 1e-2
- Batch size: 16, 32, 64
- Image size: 224x224 (standard), 384x384 (if memory allows)
- Epochs: 15-30
- Optimizer: Adam, AdamW, SGD

**Monitoring & Validation**:
```python
# Track metrics
- Training loss/accuracy
- Validation loss/accuracy
- Learning rate schedule
- Confusion matrix on validation set

# Visualize
learn.recorder.plot_loss()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9)
```

**Model Export**:
```python
# Save best model based on validation accuracy
learn.export('model.pkl')
```

---

### Stage 3: Model Evaluation & Analysis
**Notebook**: `03_model_evaluation.ipynb`

**Objectives**:
- Load trained model
- Evaluate on test set
- Generate comprehensive metrics
- Analyze errors
- Test on external dataset

**Evaluation Metrics**:

1. **Overall Performance**
   - Global accuracy
   - Precision, Recall, F1-score per class
   - Confusion matrix

2. **Per-Class Analysis**
   ```python
   # Classification report
   from sklearn.metrics import classification_report

   # Identify:
   - Which classes perform best/worst
   - Common misclassifications
   - Class-specific patterns
   ```

3. **Error Analysis**
   - Visualize top losses
   - Examine false positives/negatives
   - Identify difficult cases
   - Check for systematic errors

4. **External Validation**
   - Test on `test_second_dataset/`
   - Compare accuracy with test set
   - Assess generalization capability

**Visualization Strategy**:
```python
# 1. Confusion matrix heatmap
# 2. ROC curves per class (if using probabilities)
# 3. Correctly classified examples (3-5 per class)
# 4. Incorrectly classified examples with predictions
# 5. Class activation maps (GradCAM) - optional but insightful
```

---

## Data Pipeline Design

### DataBlock API (fastai)
```python
from fastai.vision.all import *

# Recommended approach
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # Or use saved split
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(
        size=224,
        min_scale=0.75,
        max_rotate=15.0,
        max_lighting=0.2,
        p_affine=0.75
    )
)

dls = dblock.dataloaders(path, bs=32)
```

### Alternative: Using Saved Split
```python
# Load saved split from Stage 1
split_df = pd.read_csv('data_split.csv')

# Create custom splitter
def get_split(df):
    train_idx = df[df['split'] == 'train'].index
    valid_idx = df[df['split'] == 'val'].index
    return list(train_idx), list(valid_idx)
```

---

## Model Architecture Recommendations

### Recommendation: ResNet34 with Transfer Learning

**Rationale**:
1. **Proven Track Record**: Excellent performance on medical images
2. **Efficient**: Faster training than deeper models
3. **Balanced**: Good accuracy without overfitting risk
4. **Small Dataset Friendly**: Works well with 2,500 images

**Expected Performance**:
- Train accuracy: 95-98%
- Validation accuracy: 92-95%
- Test accuracy: 92-95%
- External dataset: ≥92% (target)

### Alternative: EfficientNet-B0
```python
learn = vision_learner(dls, efficientnet_b0, metrics=[accuracy, error_rate])
```

**Rationale**:
- State-of-the-art efficiency
- Compound scaling
- Potential for higher accuracy with similar compute

---

## Performance Optimization Strategy

### 1. Baseline Model (Quick)
- ResNet34 pretrained
- Default augmentations
- 15 epochs
- Target: 85-90% accuracy

### 2. Improved Model (Iterative)
- Experiment with architectures (ResNet50, EfficientNet)
- Enhanced augmentations
- Learning rate optimization
- 25-30 epochs
- Target: 90-93% accuracy

### 3. Advanced Techniques (If needed)
- Test-time augmentation (TTA)
- Model ensembling (2-3 models)
- Progressive resizing (224 → 384)
- Mixed precision training
- Target: 93-95% accuracy

---

## Risk Mitigation

### Overfitting Prevention
1. **Data augmentation** (essential)
2. **Dropout** in final layers
3. **Weight decay** (L2 regularization)
4. **Early stopping** based on validation loss
5. **Monitor train-val gap** continuously

### Generalization Validation
1. Test on `test_second_dataset/` during development
2. If external accuracy < 90%, increase regularization
3. Compare predictions on both datasets
4. Adjust augmentation strategy if needed

---

## File Organization

```
cv-nhan/
├── Dataset and Notebook-20251115/
│   ├── dataset_leukocytes/          # Training data
│   └── test_second_dataset/         # External validation
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Stage 1
│   ├── 02_model_training.ipynb      # Stage 2
│   └── 03_model_evaluation.ipynb    # Stage 3
├── outputs/
│   ├── data_split.csv               # Train/val/test split
│   ├── model.pkl                    # Exported model
│   └── figures/                     # Plots for report
├── requirements.txt
├── CLAUDE.md
└── SYSTEM_DESIGN.md
```

---

## Success Criteria

### Minimum Requirements
- ✓ Model loads with `load_learner()` without errors
- ✓ Test accuracy ≥ 85%
- ✓ External dataset accuracy ≥ 92% (for grade 10)
- ✓ All 5 classes predicted correctly
- ✓ Confusion matrix shows balanced performance

### Excellence Indicators
- External accuracy ≥ 95%
- Per-class F1-score ≥ 0.90
- Low train-validation gap (< 3%)
- Fast inference (< 0.1s per image)
- Clear, interpretable results

---

## Next Steps

1. **Create notebook templates** with structured code
2. **Start with Stage 1**: Data preparation and EDA
3. **Quick baseline**: Train ResNet34 for 10 epochs
4. **Iterate**: Optimize based on results
5. **Validate early**: Test on external dataset frequently
6. **Document**: Capture insights for final report
