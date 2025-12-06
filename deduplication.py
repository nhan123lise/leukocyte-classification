#!/usr/bin/env python3
"""
Reusable Image Deduplication Module

This module provides tools for detecting exact and near-duplicate images
using multiple hashing algorithms:
- MD5: Exact byte-level duplicates
- pHash: Perceptual hash (DCT-based, robust to resizing/compression)
- dHash: Difference hash (gradient-based, fast)

Usage:
    from deduplication import ImageDeduplicator

    dedup = ImageDeduplicator(phash_threshold=8)
    dedup.add_dataset("dataset1", [(path, label) for path, label in images], priority=0)
    dedup.add_dataset("dataset2", [(path, label) for path, label in images], priority=1)
    dedup.compute_hashes()
    report = dedup.find_duplicates()
    unique = dedup.get_unique_images()
"""

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import imagehash
from PIL import Image
from tqdm import tqdm


class HashType(Enum):
    """Supported hash types for deduplication."""
    MD5 = "md5"
    PHASH = "phash"
    DHASH = "dhash"


@dataclass
class ImageInfo:
    """Metadata for a single image."""
    path: Path
    dataset: str
    label: str
    priority: int = 0
    md5_hash: str = ""
    phash: str = ""
    dhash: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "dataset": self.dataset,
            "label": self.label,
            "priority": self.priority,
            "md5_hash": self.md5_hash,
            "phash": self.phash,
            "dhash": self.dhash
        }


@dataclass
class DuplicateGroup:
    """A group of duplicate/similar images."""
    hash_type: HashType
    hash_value: str
    images: List[ImageInfo] = field(default_factory=list)
    hamming_distance: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hash_type": self.hash_type.value,
            "hash_value": self.hash_value,
            "hamming_distance": self.hamming_distance,
            "images": [img.to_dict() for img in self.images]
        }


@dataclass
class DeduplicationReport:
    """Summary report of deduplication results."""
    total_images: int = 0
    unique_images: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    cross_dataset_duplicates: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    kept_images: List[ImageInfo] = field(default_factory=list)
    removed_images: List[ImageInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_images": self.total_images,
                "unique_images": self.unique_images,
                "exact_duplicates": self.exact_duplicates,
                "near_duplicates": self.near_duplicates,
                "cross_dataset_duplicates": self.cross_dataset_duplicates
            },
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
            "removed_count": len(self.removed_images)
        }


def compute_md5(image_path: Path) -> str:
    """
    Compute MD5 hash of image file bytes.

    Args:
        image_path: Path to image file

    Returns:
        Hexadecimal MD5 digest string
    """
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_phash(image_path: Path, hash_size: int = 16) -> str:
    """
    Compute perceptual hash using DCT (Discrete Cosine Transform).

    Robust to: resizing, minor color changes, JPEG compression

    Args:
        image_path: Path to image file
        hash_size: Hash dimension (16 = 256-bit hash)

    Returns:
        Hexadecimal pHash string
    """
    img = Image.open(image_path)
    return str(imagehash.phash(img, hash_size=hash_size))


def compute_dhash(image_path: Path, hash_size: int = 16) -> str:
    """
    Compute difference hash based on gradient direction.

    Fast and effective for detecting resized duplicates.

    Args:
        image_path: Path to image file
        hash_size: Hash dimension

    Returns:
        Hexadecimal dHash string
    """
    img = Image.open(image_path)
    return str(imagehash.dhash(img, hash_size=hash_size))


def compute_all_hashes(args: Tuple[Path, str, str, int, int]) -> Tuple[Path, str, str, str]:
    """
    Compute all hashes for a single image (for multiprocessing).

    Args:
        args: Tuple of (path, dataset, label, priority, hash_size)

    Returns:
        Tuple of (path, md5_hash, phash, dhash)
    """
    path, dataset, label, priority, hash_size = args
    try:
        md5 = compute_md5(path)
        phash = compute_phash(path, hash_size)
        dhash = compute_dhash(path, hash_size)
        return (path, md5, phash, dhash)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return (path, "", "", "")


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hex hash strings.

    Args:
        hash1, hash2: Hexadecimal hash strings

    Returns:
        Number of differing bits
    """
    if not hash1 or not hash2:
        return float('inf')
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


class ImageDeduplicator:
    """
    Reusable image deduplication engine.

    Supports multiple hashing algorithms and configurable thresholds
    for detecting exact and near-duplicate images.

    Example usage:
        deduplicator = ImageDeduplicator(phash_threshold=8)
        deduplicator.add_dataset("original", images_list, priority=0)
        deduplicator.add_dataset("external", external_images, priority=1)
        deduplicator.compute_hashes()
        report = deduplicator.find_duplicates()
        unique_images = deduplicator.get_unique_images()
    """

    def __init__(
        self,
        phash_threshold: int = 8,
        dhash_threshold: int = 8,
        hash_size: int = 16,
        use_md5: bool = True,
        use_phash: bool = True,
        use_dhash: bool = True,
        n_workers: int = 4
    ):
        """
        Initialize deduplicator with configurable thresholds.

        Args:
            phash_threshold: Max Hamming distance for pHash similarity (0=exact, 8=similar)
            dhash_threshold: Max Hamming distance for dHash similarity
            hash_size: Size of perceptual hash (default 16 = 256-bit hash)
            use_md5: Enable exact duplicate detection
            use_phash: Enable perceptual hash comparison
            use_dhash: Enable difference hash comparison
            n_workers: Number of parallel workers for hashing
        """
        self.phash_threshold = phash_threshold
        self.dhash_threshold = dhash_threshold
        self.hash_size = hash_size
        self.use_md5 = use_md5
        self.use_phash = use_phash
        self.use_dhash = use_dhash
        self.n_workers = n_workers

        self.images: List[ImageInfo] = []
        self.datasets: Dict[str, int] = {}  # name -> priority
        self._hashes_computed = False
        self._report: Optional[DeduplicationReport] = None

    def add_dataset(
        self,
        name: str,
        images: List[Tuple[Path, str]],
        priority: int = 0
    ) -> None:
        """
        Add a dataset to the deduplication pool.

        Args:
            name: Dataset identifier (e.g., "original", "pbc_dataset")
            images: List of (filepath, label) tuples
            priority: Lower = keep preferentially when deduplicating
                     (original dataset should have priority=0)
        """
        self.datasets[name] = priority
        for path, label in images:
            self.images.append(ImageInfo(
                path=Path(path),
                dataset=name,
                label=label,
                priority=priority
            ))
        print(f"  Added dataset '{name}': {len(images)} images (priority={priority})")

    def compute_hashes(self, show_progress: bool = True) -> None:
        """
        Compute all enabled hashes for all loaded images.
        Uses multiprocessing for parallel computation.
        """
        if self._hashes_computed:
            print("  Hashes already computed, skipping...")
            return

        print(f"  Computing hashes for {len(self.images)} images...")
        print(f"  Using {self.n_workers} workers...")

        # Prepare args for multiprocessing
        args_list = [
            (img.path, img.dataset, img.label, img.priority, self.hash_size)
            for img in self.images
        ]

        # Create a mapping from path to image index
        path_to_idx = {img.path: idx for idx, img in enumerate(self.images)}

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(compute_all_hashes, args): args[0]
                      for args in args_list}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="  Hashing")

            for future in iterator:
                path, md5, phash, dhash = future.result()
                idx = path_to_idx[path]
                self.images[idx].md5_hash = md5
                self.images[idx].phash = phash
                self.images[idx].dhash = dhash

        self._hashes_computed = True
        print(f"  ✓ Computed hashes for {len(self.images)} images")

    def find_duplicates(self) -> DeduplicationReport:
        """
        Find all duplicate and near-duplicate images.

        Returns:
            DeduplicationReport with detailed findings
        """
        if not self._hashes_computed:
            raise RuntimeError("Must call compute_hashes() before find_duplicates()")

        report = DeduplicationReport(total_images=len(self.images))

        # Track which images are duplicates
        duplicate_indices: Set[int] = set()

        # 1. Find exact MD5 duplicates
        if self.use_md5:
            md5_groups = defaultdict(list)
            for idx, img in enumerate(self.images):
                if img.md5_hash:
                    md5_groups[img.md5_hash].append(idx)

            for hash_val, indices in md5_groups.items():
                if len(indices) > 1:
                    group = DuplicateGroup(
                        hash_type=HashType.MD5,
                        hash_value=hash_val,
                        images=[self.images[i] for i in indices],
                        hamming_distance=0
                    )
                    report.duplicate_groups.append(group)
                    report.exact_duplicates += len(indices) - 1

                    # Check if cross-dataset
                    datasets_in_group = set(self.images[i].dataset for i in indices)
                    if len(datasets_in_group) > 1:
                        report.cross_dataset_duplicates += len(indices) - 1

                    # Mark all but the highest priority as duplicates
                    sorted_indices = sorted(indices, key=lambda i: self.images[i].priority)
                    for i in sorted_indices[1:]:
                        duplicate_indices.add(i)

        # 2. Find near-duplicates using pHash
        if self.use_phash:
            self._find_perceptual_duplicates(
                report, duplicate_indices,
                hash_attr='phash',
                threshold=self.phash_threshold,
                hash_type=HashType.PHASH
            )

        # 3. Find near-duplicates using dHash
        if self.use_dhash:
            self._find_perceptual_duplicates(
                report, duplicate_indices,
                hash_attr='dhash',
                threshold=self.dhash_threshold,
                hash_type=HashType.DHASH
            )

        # Calculate unique count
        report.unique_images = len(self.images) - len(duplicate_indices)

        # Separate kept and removed
        for idx, img in enumerate(self.images):
            if idx in duplicate_indices:
                report.removed_images.append(img)
            else:
                report.kept_images.append(img)

        self._report = report
        return report

    def _find_perceptual_duplicates(
        self,
        report: DeduplicationReport,
        duplicate_indices: Set[int],
        hash_attr: str,
        threshold: int,
        hash_type: HashType
    ) -> None:
        """Find near-duplicates using perceptual hashing."""
        # Group by similar hashes using union-find approach
        n = len(self.images)

        # For efficiency, first group by hash prefix (first 4 chars)
        prefix_groups = defaultdict(list)
        for idx, img in enumerate(self.images):
            hash_val = getattr(img, hash_attr)
            if hash_val:
                prefix = hash_val[:4]
                prefix_groups[prefix].append(idx)

        # Check within and across prefix groups
        checked_pairs: Set[Tuple[int, int]] = set()
        near_dup_groups: List[DuplicateGroup] = []

        # Get all hashes
        all_hashes = [getattr(img, hash_attr) for img in self.images]

        # Compare images - use a more efficient approach
        for prefix, indices in prefix_groups.items():
            if len(indices) <= 1:
                continue

            for i, idx1 in enumerate(indices):
                for idx2 in indices[i+1:]:
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    # Already marked as exact duplicate
                    if idx1 in duplicate_indices and idx2 in duplicate_indices:
                        continue

                    dist = hamming_distance(all_hashes[idx1], all_hashes[idx2])
                    if dist <= threshold and dist > 0:  # > 0 to exclude exact matches
                        group = DuplicateGroup(
                            hash_type=hash_type,
                            hash_value=f"{all_hashes[idx1]}~{all_hashes[idx2]}",
                            images=[self.images[idx1], self.images[idx2]],
                            hamming_distance=dist
                        )
                        near_dup_groups.append(group)
                        report.near_duplicates += 1

                        # Check cross-dataset
                        if self.images[idx1].dataset != self.images[idx2].dataset:
                            report.cross_dataset_duplicates += 1

                        # Mark lower priority as duplicate
                        if self.images[idx1].priority <= self.images[idx2].priority:
                            duplicate_indices.add(idx2)
                        else:
                            duplicate_indices.add(idx1)

        report.duplicate_groups.extend(near_dup_groups)

    def get_unique_images(
        self,
        prefer_dataset: Optional[str] = None,
        exclude_datasets: Optional[List[str]] = None
    ) -> List[ImageInfo]:
        """
        Get list of unique images after deduplication.

        Args:
            prefer_dataset: Prefer images from this dataset when resolving duplicates
            exclude_datasets: List of dataset names to exclude from results

        Returns:
            List of ImageInfo for unique images
        """
        if self._report is None:
            raise RuntimeError("Must call find_duplicates() before get_unique_images()")

        unique = self._report.kept_images

        if exclude_datasets:
            unique = [img for img in unique if img.dataset not in exclude_datasets]

        return unique

    def generate_report(
        self,
        output_path: Path,
        include_details: bool = True
    ) -> None:
        """
        Generate detailed deduplication report as JSON.

        Args:
            output_path: Path to save JSON report
            include_details: Include full duplicate group details
        """
        if self._report is None:
            raise RuntimeError("Must call find_duplicates() before generate_report()")

        report_dict = self._report.to_dict()

        # Add dataset statistics
        report_dict["by_dataset"] = {}
        for dataset_name in self.datasets:
            loaded = sum(1 for img in self.images if img.dataset == dataset_name)
            kept = sum(1 for img in self._report.kept_images if img.dataset == dataset_name)
            removed = sum(1 for img in self._report.removed_images if img.dataset == dataset_name)
            report_dict["by_dataset"][dataset_name] = {
                "loaded": loaded,
                "kept": kept,
                "removed": removed
            }

        # Add settings
        report_dict["settings"] = {
            "phash_threshold": self.phash_threshold,
            "dhash_threshold": self.dhash_threshold,
            "hash_size": self.hash_size,
            "use_md5": self.use_md5,
            "use_phash": self.use_phash,
            "use_dhash": self.use_dhash
        }

        if not include_details:
            report_dict.pop("duplicate_groups", None)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"  ✓ Report saved to: {output_path}")


def quick_dedup(
    image_paths: List[Path],
    phash_threshold: int = 8
) -> Tuple[List[Path], List[Path]]:
    """
    Quick utility function for simple deduplication.

    Args:
        image_paths: List of image paths to deduplicate
        phash_threshold: Similarity threshold

    Returns:
        Tuple of (unique_paths, duplicate_paths)
    """
    dedup = ImageDeduplicator(phash_threshold=phash_threshold)
    images = [(p, "default") for p in image_paths]
    dedup.add_dataset("default", images, priority=0)
    dedup.compute_hashes()
    report = dedup.find_duplicates()

    unique = [img.path for img in report.kept_images]
    duplicates = [img.path for img in report.removed_images]

    return unique, duplicates


if __name__ == "__main__":
    # Test with sample images
    print("=" * 70)
    print("DEDUPLICATION MODULE TEST")
    print("=" * 70)

    # Quick test
    from pathlib import Path
    test_path = Path("Dataset and Notebook-20251115/dataset_leukocytes/basophil")

    if test_path.exists():
        images = list(test_path.glob("*.jpg"))[:10]
        print(f"\nTesting with {len(images)} images from {test_path}")

        dedup = ImageDeduplicator(phash_threshold=8, n_workers=2)
        dedup.add_dataset("test", [(p, "basophil") for p in images], priority=0)
        dedup.compute_hashes()
        report = dedup.find_duplicates()

        print(f"\nResults:")
        print(f"  Total: {report.total_images}")
        print(f"  Unique: {report.unique_images}")
        print(f"  Exact duplicates: {report.exact_duplicates}")
        print(f"  Near duplicates: {report.near_duplicates}")
    else:
        print(f"Test path not found: {test_path}")
