"""
GPSBench Tinker Fine-tuning Package.

This package provides Tinker-based fine-tuning for GPSBench coordinate tasks.
Based on the Tinker Cookbook supervised learning recipes.
"""

from .gpsbench_datasets import (
    GPSBenchDataset,
    GPSBenchHFDataset,
    load_gpsbench_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "GPSBenchDataset",
    "GPSBenchHFDataset",
    "load_gpsbench_dataset",
]
