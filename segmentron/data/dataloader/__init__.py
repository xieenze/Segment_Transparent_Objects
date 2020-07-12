"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .trans10k import TransSegmentation
from .trans10k_boundary import TransSegmentationBoundary
from .trans10k_extra import TransExtraSegmentation

datasets = {
    'trans10k': TransSegmentation,
    'trans10k_boundary': TransSegmentationBoundary,
    'trans10k_extra': TransExtraSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
