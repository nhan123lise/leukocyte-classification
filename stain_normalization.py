"""
Stain Normalization Module for Histopathology Images

Implements Reinhard color normalization to standardize staining appearance
across different microscopy protocols.

Reference: Reinhard et al., "Color Transfer between Images" (2001)
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space."""
    # Normalize to [0, 1]
    rgb = rgb.astype(np.float64) / 255.0

    # RGB to XYZ
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # sRGB D65 illuminant
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    xyz = rgb @ xyz_matrix.T

    # XYZ to LAB
    # D65 reference white
    ref_white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / ref_white

    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1/3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16/116)

    L = (116 * xyz[..., 1]) - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB image to RGB color space."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    xyz = np.stack([x, y, z], axis=-1)

    mask = xyz > 0.2068966
    xyz[mask] = xyz[mask] ** 3
    xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

    # D65 reference white
    ref_white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz * ref_white

    # XYZ to RGB
    rgb_matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])

    rgb = xyz @ rgb_matrix.T

    # Apply gamma correction
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    # Clip and convert to uint8
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb


class ReinhardNormalizer:
    """
    Reinhard color normalization for stain standardization.

    Transfers the color distribution of source images to match a target image
    using statistics in LAB color space.
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None
        self.fitted = False

    def fit(self, target_image: Union[np.ndarray, Image.Image, str, Path]) -> 'ReinhardNormalizer':
        """
        Fit normalizer to a target/reference image.

        Args:
            target_image: Reference image (array, PIL Image, or path)

        Returns:
            self for chaining
        """
        # Load image if path
        if isinstance(target_image, (str, Path)):
            target_image = np.array(Image.open(target_image).convert('RGB'))
        elif isinstance(target_image, Image.Image):
            target_image = np.array(target_image.convert('RGB'))

        # Convert to LAB
        lab = rgb_to_lab(target_image)

        # Compute statistics per channel
        self.target_means = np.mean(lab, axis=(0, 1))
        self.target_stds = np.std(lab, axis=(0, 1))

        # Prevent division by zero
        self.target_stds = np.maximum(self.target_stds, 1e-6)

        self.fitted = True
        return self

    def fit_to_stats(
        self,
        means: Tuple[float, float, float],
        stds: Tuple[float, float, float]
    ) -> 'ReinhardNormalizer':
        """
        Fit normalizer to pre-computed LAB statistics.

        Args:
            means: (L_mean, a_mean, b_mean)
            stds: (L_std, a_std, b_std)

        Returns:
            self for chaining
        """
        self.target_means = np.array(means)
        self.target_stds = np.array(stds)
        self.target_stds = np.maximum(self.target_stds, 1e-6)
        self.fitted = True
        return self

    def transform(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        return_pil: bool = False
    ) -> Union[np.ndarray, Image.Image]:
        """
        Normalize an image to match the target color distribution.

        Args:
            image: Image to normalize
            return_pil: If True, return PIL Image instead of numpy array

        Returns:
            Normalized image
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        # Load image if path
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        # Convert to LAB
        lab = rgb_to_lab(image)

        # Compute source statistics
        source_means = np.mean(lab, axis=(0, 1))
        source_stds = np.std(lab, axis=(0, 1))
        source_stds = np.maximum(source_stds, 1e-6)

        # Normalize: (x - src_mean) / src_std * tgt_std + tgt_mean
        lab = (lab - source_means) / source_stds * self.target_stds + self.target_means

        # Convert back to RGB
        rgb = lab_to_rgb(lab)

        if return_pil:
            return Image.fromarray(rgb)
        return rgb

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        return_pil: bool = False
    ) -> Union[np.ndarray, Image.Image]:
        """Alias for transform()."""
        return self.transform(image, return_pil)


# Pre-computed statistics from training data (pink staining)
# Computed from 500 samples of dataset_leukocytes images
TRAINING_DATA_STATS = {
    'means': (78.8, 11.3, 6.7),   # L, a, b means
    'stds': (2.3, 1.4, 2.9)       # L, a, b standard deviations
}

# Statistics from external 'dataset' source (golden/purple staining)
# Used for stain augmentation during training
DATASET_SOURCE_STATS = {
    'means': (62.0, 16.9, -2.5),  # L, a, b means (darker, more purple)
    'stds': (5.0, 4.4, 5.2)       # L, a, b standard deviations
}


def get_default_normalizer() -> ReinhardNormalizer:
    """
    Get a normalizer fitted to the training data statistics.

    Returns:
        ReinhardNormalizer fitted to training distribution
    """
    normalizer = ReinhardNormalizer()
    normalizer.fit_to_stats(
        means=TRAINING_DATA_STATS['means'],
        stds=TRAINING_DATA_STATS['stds']
    )
    return normalizer


def compute_dataset_stats(image_paths: list, sample_size: int = 100) -> dict:
    """
    Compute LAB statistics for a dataset.

    Args:
        image_paths: List of image paths
        sample_size: Number of images to sample (for efficiency)

    Returns:
        Dictionary with 'means' and 'stds'
    """
    import random

    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)

    all_L, all_a, all_b = [], [], []

    for path in image_paths:
        try:
            img = np.array(Image.open(path).convert('RGB'))
            lab = rgb_to_lab(img)
            all_L.append(lab[..., 0].mean())
            all_a.append(lab[..., 1].mean())
            all_b.append(lab[..., 2].mean())
        except Exception:
            continue

    return {
        'means': (np.mean(all_L), np.mean(all_a), np.mean(all_b)),
        'stds': (np.std(all_L), np.std(all_a), np.std(all_b))
    }


# FastAI-compatible stain augmentation for training
class StainAugment:
    """
    Randomly transform images to different staining styles during training.

    This helps the model learn features that are invariant to staining protocol.

    Usage:
        augment = StainAugment(p=0.5)
        dblock = DataBlock(
            ...
            item_tfms=[augment, Resize(224)],
            ...
        )
    """

    def __init__(self, p: float = 0.5, interpolate: bool = True):
        """
        Args:
            p: Probability of applying stain augmentation
            interpolate: If True, interpolate between original and target style
                        (more diverse augmentation). If False, full transfer.
        """
        self.p = p
        self.interpolate = interpolate
        self.target_normalizer = ReinhardNormalizer()
        self.target_normalizer.fit_to_stats(
            means=DATASET_SOURCE_STATS['means'],
            stds=DATASET_SOURCE_STATS['stds']
        )

    def __call__(self, img):
        """Apply random stain augmentation to a PIL Image."""
        import random
        from PIL import ImageEnhance

        if not isinstance(img, Image.Image):
            return img

        if random.random() > self.p:
            return img

        # Transform to target staining style (color transfer)
        transformed = self.target_normalizer.transform(img, return_pil=False)

        if self.interpolate:
            # Random interpolation between original and transformed
            alpha = random.uniform(0.3, 1.0)
            original = np.array(img)
            blended = (alpha * transformed + (1 - alpha) * original).astype(np.uint8)
            result = Image.fromarray(blended)
        else:
            result = Image.fromarray(transformed)

        # Also reduce saturation to match "dataset" source (~50% of training)
        # Dataset has saturation 0.285 vs training 0.600 = about 47%
        sat_factor = random.uniform(0.4, 0.6)  # Reduce saturation to 40-60%
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(sat_factor)

        return result


# FastAI-compatible transform
class StainNormalize:
    """
    FastAI-compatible stain normalization transform.

    Usage:
        normalizer = StainNormalize()
        dblock = DataBlock(
            ...
            item_tfms=[normalizer, Resize(224)],
            ...
        )
    """

    def __init__(self, normalizer: Optional[ReinhardNormalizer] = None):
        """
        Args:
            normalizer: Pre-fitted normalizer, or None to use default
        """
        self.normalizer = normalizer or get_default_normalizer()

    def __call__(self, img):
        """Apply stain normalization to a PIL Image."""
        if isinstance(img, Image.Image):
            return self.normalizer.transform(img, return_pil=True)
        return img


if __name__ == "__main__":
    # Demo: compute stats from training data
    from pathlib import Path

    train_path = Path('/Users/mac/code/cv-nhan/Dataset and Notebook-20251115/dataset_leukocytes')
    if train_path.exists():
        image_paths = list(train_path.glob('*/*.jpg'))
        stats = compute_dataset_stats(image_paths, sample_size=200)
        print("Training data LAB statistics:")
        print(f"  Means (L, a, b): {stats['means']}")
        print(f"  Stds (L, a, b): {stats['stds']}")
