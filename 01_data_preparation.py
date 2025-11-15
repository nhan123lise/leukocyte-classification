#!/usr/bin/env python3
"""
Stage 1: Data Preparation & Exploration

This script:
- Loads and explores the leukocyte dataset
- Verifies class distribution (500 images per class)
- Performs exploratory data analysis (EDA)
- Creates stratified train/validation/test split
- Saves split configuration for reproducibility

Output: data_split.csv with file paths and split labels
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from utils import set_seed

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Set random seed for reproducibility across all libraries
SEED = 42
set_seed(SEED)

# Define split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("=" * 70)
print("LEUKOCYTE DATASET PREPARATION")
print("=" * 70)


def create_dataset_dataframe(base_path):
    """
    Create a dataframe with image paths and labels
    """
    data = []

    # Expected classes
    classes = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

    for class_name in classes:
        class_path = base_path / class_name
        if class_path.exists():
            # Get all jpg files
            image_files = list(class_path.glob('*.jpg'))

            for img_path in image_files:
                data.append({
                    'filepath': str(img_path),
                    'filename': img_path.name,
                    'label': class_name
                })

    return pd.DataFrame(data)


def get_image_dimensions(filepath):
    """Get image dimensions"""
    with Image.open(filepath) as img:
        return img.size  # (width, height)


def main():
    # 1. Load Dataset
    print("\n1. LOADING DATASET")
    print("-" * 70)

    dataset_path = Path('Dataset and Notebook-20251115/dataset_leukocytes')

    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    print(f"Dataset path: {dataset_path}")

    # Create dataframe
    df = create_dataset_dataframe(dataset_path)

    print(f"✓ Total images: {len(df)}")
    print(f"✓ First few rows:")
    print(df.head())

    # 2. Class Distribution
    print("\n2. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 70)

    class_counts = df['label'].value_counts().sort_index()
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        status = "✓" if count == 500 else "✗"
        print(f"  {status} {class_name:12s}: {count:4d} images")

    print(f"\nExpected: 500 images per class")
    print(f"Total: {len(df)} images")

    # Visualize class distribution
    os.makedirs('outputs/figures', exist_ok=True)

    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Leukocyte Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(y=500, color='r', linestyle='--', label='Expected (500)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: outputs/figures/class_distribution.png")
    plt.close()

    # 3. Image Dimensions Analysis
    print("\n3. IMAGE DIMENSIONS ANALYSIS")
    print("-" * 70)

    # Sample 100 random images to check dimensions
    sample_df = df.sample(n=min(100, len(df)), random_state=SEED)
    dimensions = [get_image_dimensions(fp) for fp in sample_df['filepath']]

    widths = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]

    print(f"\nImage dimension statistics (sample of {len(dimensions)}):")
    print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.2f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.2f}")

    # Check if all images have same dimensions
    unique_dims = set(dimensions)
    print(f"\nUnique dimensions found: {len(unique_dims)}")
    if len(unique_dims) <= 5:
        for dim in unique_dims:
            print(f"  {dim}")

    # 4. Visualize Sample Images
    print("\n4. VISUALIZING SAMPLE IMAGES")
    print("-" * 70)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Sample Images from Each Leukocyte Class', fontsize=16, fontweight='bold')

    classes = sorted(df['label'].unique())

    for idx, class_name in enumerate(classes):
        # Get 5 random samples from this class
        class_samples = df[df['label'] == class_name].sample(n=5, random_state=SEED)

        for col, (_, row) in enumerate(class_samples.iterrows()):
            img = Image.open(row['filepath'])
            axes[idx, col].imshow(img)
            axes[idx, col].axis('off')

            if col == 0:
                axes[idx, col].set_title(class_name.capitalize(), fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig('outputs/figures/sample_images.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: outputs/figures/sample_images.png")
    plt.close()

    # 5. Create Train/Validation/Test Split
    print("\n5. CREATING DATA SPLIT")
    print("-" * 70)

    print(f"\nSplit ratios:")
    print(f"  Train: {TRAIN_RATIO*100}%")
    print(f"  Validation: {VAL_RATIO*100}%")
    print(f"  Test: {TEST_RATIO*100}%")

    # Perform stratified split
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df['label'],
        random_state=SEED
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df['label'],
        random_state=SEED
    )

    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine back
    df_split = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_df):4d} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):4d} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):4d} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total:      {len(df_split):4d}")

    # 6. Verify Stratification
    print("\n6. VERIFYING STRATIFICATION")
    print("-" * 70)

    print("\nClass distribution per split:")
    split_distribution = df_split.groupby(['split', 'label']).size().unstack(fill_value=0)
    print(split_distribution)

    # Visualize split distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Class Distribution Across Splits', fontsize=16, fontweight='bold')

    for idx, split in enumerate(['train', 'val', 'test']):
        split_data = df_split[df_split['split'] == split]['label'].value_counts().sort_index()
        split_data.plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')
        axes[idx].set_title(f'{split.capitalize()} Set ({len(df_split[df_split["split"] == split])} images)',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Class', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('outputs/figures/split_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: outputs/figures/split_distribution.png")
    plt.close()

    # 7. Save Split Configuration
    print("\n7. SAVING SPLIT CONFIGURATION")
    print("-" * 70)

    output_path = 'outputs/data_split.csv'
    df_split.to_csv(output_path, index=False)
    print(f"✓ Split configuration saved to: {output_path}")

    # Verify saved file
    df_verify = pd.read_csv(output_path)
    print(f"\nVerification:")
    print(f"  Rows in saved file: {len(df_verify)}")
    print(f"  Columns: {list(df_verify.columns)}")

    # 8. Summary
    print("\n" + "=" * 70)
    print("DATA PREPARATION SUMMARY")
    print("=" * 70)
    print(f"\nDataset Information:")
    print(f"  Total images: {len(df_split)}")
    print(f"  Number of classes: {df_split['label'].nunique()}")
    print(f"  Classes: {sorted(df_split['label'].unique())}")
    print(f"\nSplit Configuration:")
    print(f"  Train: {len(train_df)} images ({len(train_df)/len(df_split)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} images ({len(val_df)/len(df_split)*100:.1f}%)")
    print(f"  Test: {len(test_df)} images ({len(test_df)/len(df_split)*100:.1f}%)")
    print(f"\nOutput Files:")
    print(f"  ✓ {output_path}")
    print(f"  ✓ outputs/figures/class_distribution.png")
    print(f"  ✓ outputs/figures/sample_images.png")
    print(f"  ✓ outputs/figures/split_distribution.png")
    print("\n" + "=" * 70)
    print("✓ Data preparation complete!")
    print("Next: Open and run notebooks/02_model_training.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
