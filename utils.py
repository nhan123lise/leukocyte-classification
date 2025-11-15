"""
Utility functions for the leukocyte classification project
"""

import random
import numpy as np
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries

    Args:
        seed (int): Random seed value (default: 42)
    """
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        # Make PyTorch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    except ImportError:
        pass

    print(f"✓ Random seed set to {seed} for reproducibility")


def get_project_root():
    """Get the project root directory"""
    from pathlib import Path
    return Path(__file__).parent


def verify_dataset(dataset_path, expected_classes=None, expected_per_class=500):
    """
    Verify dataset structure and class distribution

    Args:
        dataset_path (Path): Path to dataset directory
        expected_classes (list): List of expected class names
        expected_per_class (int): Expected number of images per class

    Returns:
        bool: True if dataset is valid
    """
    from pathlib import Path

    if expected_classes is None:
        expected_classes = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        return False

    all_valid = True
    for class_name in expected_classes:
        class_path = dataset_path / class_name
        if not class_path.exists():
            print(f"✗ Class directory missing: {class_name}")
            all_valid = False
            continue

        images = list(class_path.glob('*.jpg'))
        if len(images) != expected_per_class:
            print(f"✗ {class_name}: Expected {expected_per_class}, found {len(images)} images")
            all_valid = False
        else:
            print(f"✓ {class_name}: {len(images)} images")

    return all_valid


if __name__ == "__main__":
    # Test seed setting
    set_seed(42)

    # Test dataset verification
    print("\nVerifying dataset...")
    from pathlib import Path
    dataset_path = Path("Dataset and Notebook-20251115/dataset_leukocytes")
    verify_dataset(dataset_path)
