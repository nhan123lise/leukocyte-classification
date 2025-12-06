"""
Stain Normalization Module for Histopathology/Cytology Images

Uses torchstain for GPU-accelerated Macenko and Reinhard normalization.

Methods:
1. Macenko (2009) - SVD-based stain separation, best for H&E/Giemsa staining
2. Reinhard (2001) - LAB color space transfer, general purpose
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional

from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
from torchstain.torch.normalizers.reinhard import TorchReinhardNormalizer


def _load_image(image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
    """Load image as numpy array."""
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert('RGB'))
    elif isinstance(image, Image.Image):
        return np.array(image.convert('RGB'))
    return image


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert HWC numpy array to CHW tensor."""
    return torch.from_numpy(img).permute(2, 0, 1).float()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to HWC numpy array."""
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        return tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    return tensor.numpy().astype(np.uint8)


class MacenkoNormalizer:
    """Macenko stain normalization using torchstain."""

    def __init__(self):
        self._normalizer = TorchMacenkoNormalizer()
        self.fitted = False

    def fit(self, target_image: Union[np.ndarray, Image.Image, str, Path]) -> 'MacenkoNormalizer':
        """Fit normalizer to a reference image."""
        img = _load_image(target_image)
        self._normalizer.fit(_to_tensor(img))
        self.fitted = True
        return self

    def transform(self, image: Union[np.ndarray, Image.Image, str, Path], return_pil: bool = False):
        """Normalize an image to match the reference."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        img = _load_image(image)
        result, _, _ = self._normalizer.normalize(I=_to_tensor(img), stains=False)
        result = _to_numpy(result)
        return Image.fromarray(result) if return_pil else result

    def __call__(self, image, return_pil: bool = False):
        return self.transform(image, return_pil)


class ReinhardNormalizer:
    """Reinhard color normalization using torchstain."""

    def __init__(self):
        self._normalizer = TorchReinhardNormalizer()
        self.fitted = False

    def fit(self, target_image: Union[np.ndarray, Image.Image, str, Path]) -> 'ReinhardNormalizer':
        """Fit normalizer to a reference image."""
        img = _load_image(target_image)
        self._normalizer.fit(_to_tensor(img))
        self.fitted = True
        return self

    def transform(self, image: Union[np.ndarray, Image.Image, str, Path], return_pil: bool = False):
        """Normalize an image to match the target color distribution."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        img = _load_image(image)
        result = self._normalizer.normalize(I=_to_tensor(img))
        result = _to_numpy(result)
        return Image.fromarray(result) if return_pil else result

    def __call__(self, image, return_pil: bool = False):
        return self.transform(image, return_pil)


class StainAugment:
    """
    Stain augmentation for training.

    Transforms images toward a different staining style with random intensity.
    """

    def __init__(self,
                 p: float = 0.5,
                 reference_image: Union[str, Path, np.ndarray, Image.Image] = None,
                 intensity_range: Tuple[float, float] = (0.3, 1.0)):
        self.p = p
        self.intensity_range = intensity_range
        self.normalizer = None

        if reference_image is not None:
            self.normalizer = MacenkoNormalizer()
            self.normalizer.fit(reference_image)

    def __call__(self, img):
        import random

        if not isinstance(img, Image.Image):
            return img

        if random.random() > self.p or self.normalizer is None:
            return img

        try:
            transformed = self.normalizer.transform(img, return_pil=False)
            alpha = random.uniform(*self.intensity_range)
            original = np.array(img)
            blended = (alpha * transformed + (1 - alpha) * original).astype(np.uint8)
            return Image.fromarray(blended)
        except Exception:
            return img


def get_normalizer(method: str, reference_image: Union[str, Path]):
    """Get a fitted normalizer."""
    normalizer = MacenkoNormalizer() if method == 'macenko' else ReinhardNormalizer()
    normalizer.fit(reference_image)
    return normalizer


if __name__ == "__main__":
    print("Stain Normalization Module (torchstain)")
    print("=" * 50)

    train_path = Path('/Users/mac/code/cv-nhan/Dataset and Notebook-20251115/dataset_leukocytes')
    if train_path.exists():
        images = list(train_path.glob('*/*.jpg'))

        print("\nTesting Macenko...")
        macenko = MacenkoNormalizer()
        macenko.fit(images[0])
        result = macenko.transform(images[100])
        print(f"  Output: {result.shape}")

        print("\nTesting Reinhard...")
        reinhard = ReinhardNormalizer()
        reinhard.fit(images[0])
        result = reinhard.transform(images[100])
        print(f"  Output: {result.shape}")

        print("\n✓ All tests passed!")
