"""
DOTpy: Deconvolution by Optimal Transport for spatial transcriptomics

A PyTorch implementation of DOT for transferring cell type annotations from 
single-cell RNA-seq to spatial transcriptomics data using multi-objective 
optimization with Frank-Wolfe algorithm.
"""

from .core import DOT
from .preprocessing import setup_reference, setup_spatial
from .visualization import plot_spatial_weights

__version__ = "0.1.0"
__all__ = ["DOT", "setup_reference", "setup_spatial", "plot_spatial_weights"]