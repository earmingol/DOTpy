"""
Unit tests for DOTpy
"""

import pytest
import numpy as np
import torch
from anndata import AnnData
from dotpy import setup_reference, setup_spatial, DOT


@pytest.fixture
def synthetic_reference():
    """Create synthetic reference data."""
    n_cells = 500
    n_genes = 100
    
    adata = AnnData(
        X=np.random.negative_binomial(5, 0.3, (n_cells, n_genes)).astype(float)
    )
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs['cell_type'] = np.random.choice(
        ['TypeA', 'TypeB', 'TypeC'],
        size=n_cells
    )
    return adata


@pytest.fixture
def synthetic_spatial():
    """Create synthetic spatial data."""
    n_spots = 100
    n_genes = 100
    
    adata = AnnData(
        X=np.random.negative_binomial(5, 0.3, (n_spots, n_genes)).astype(float)
    )
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Grid coordinates
    grid_size = 10
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.flatten(), yy.flatten()])[:n_spots]
    adata.obsm['spatial'] = coords
    
    return adata


def test_setup_reference(synthetic_reference):
    """Test reference data processing."""
    ref = setup_reference(
        synthetic_reference,
        cell_type_key='cell_type',
        subcluster_size=2,
        max_genes=50,
        device='cpu'
    )
    
    assert 'X' in ref
    assert 'clusters' in ref
    assert 'ratios' in ref
    assert isinstance(ref['X'], torch.Tensor)
    assert ref['X'].shape[1] <= 50  # max_genes


def test_setup_spatial(synthetic_spatial):
    """Test spatial data processing."""
    spatial = setup_spatial(
        synthetic_spatial,
        spatial_key='spatial',
        th_spatial=0.8,
        device='cpu'
    )
    
    assert 'X' in spatial
    assert 'coords' in spatial
    assert isinstance(spatial['X'], torch.Tensor)
    assert spatial['coords'].shape[1] == 2


def test_dot_initialization(synthetic_reference, synthetic_spatial):
    """Test DOT object initialization."""
    ref = setup_reference(
        synthetic_reference,
        cell_type_key='cell_type',
        device='cpu'
    )
    spatial = setup_spatial(
        synthetic_spatial,
        spatial_key='spatial',
        device='cpu'
    )
    
    dot = DOT(spatial, ref, ls_solution=True)
    
    assert dot.solution is not None
    assert dot.ref is not None
    assert dot.spatial is not None


def test_dot_fit(synthetic_reference, synthetic_spatial):
    """Test DOT fitting."""
    ref = setup_reference(
        synthetic_reference,
        cell_type_key='cell_type',
        subcluster_size=2,
        max_genes=50,
        device='cpu'
    )
    spatial = setup_spatial(
        synthetic_spatial,
        spatial_key='spatial',
        th_spatial=0.8,
        device='cpu'
    )
    
    dot = DOT(spatial, ref)
    dot.fit(
        mode='highres',
        iterations=5,  # Few iterations for testing
        verbose=False
    )
    
    assert dot.weights is not None
    assert dot.history is not None
    
    weights = dot.get_weights()
    assert weights.shape[0] == 100  # n_spots
    assert weights.shape[1] == 3   # n_cell_types


def test_get_weights_normalize(synthetic_reference, synthetic_spatial):
    """Test weight normalization."""
    ref = setup_reference(
        synthetic_reference,
        cell_type_key='cell_type',
        device='cpu'
    )
    spatial = setup_spatial(
        synthetic_spatial,
        spatial_key='spatial',
        device='cpu'
    )
    
    dot = DOT(spatial, ref)
    dot.fit(mode='highres', iterations=5, verbose=False)
    
    weights_norm = dot.get_weights(normalize=True)
    
    # Check normalization
    row_sums = weights_norm.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)), decimal=5)


def test_gpu_if_available(synthetic_reference, synthetic_spatial):
    """Test GPU functionality if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    ref = setup_reference(
        synthetic_reference,
        cell_type_key='cell_type',
        device='cuda'
    )
    spatial = setup_spatial(
        synthetic_spatial,
        spatial_key='spatial',
        device='cuda'
    )
    
    assert ref['X'].is_cuda
    assert spatial['X'].is_cuda
    
    dot = DOT(spatial, ref)
    dot.fit(mode='highres', iterations=5, verbose=False)
    
    assert dot.weights.is_cuda
    
    # Move to CPU for checking
    weights = dot.get_weights()
    assert isinstance(weights, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
