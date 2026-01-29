"""
Example usage of DOTpy

This script demonstrates how to use DOTr for spatial transcriptomics deconvolution.
"""

import numpy as np
import scanpy as sc
import torch
from dotpy import setup_reference, setup_spatial, DOT, plot_spatial_weights


def example_synthetic_data():
    """Create synthetic example data for testing."""
    print("Generating synthetic data...")
    
    # Create synthetic reference scRNA-seq
    n_cells = 2000
    n_genes = 500
    n_cell_types = 5
    
    ref_adata = sc.AnnData(
        X=np.random.negative_binomial(5, 0.3, (n_cells, n_genes)).astype(float)
    )
    ref_adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    ref_adata.obs['cell_type'] = np.random.choice(
        [f'CellType_{i}' for i in range(n_cell_types)],
        size=n_cells
    )
    
    # Create synthetic spatial data
    n_spots = 500
    spatial_adata = sc.AnnData(
        X=np.random.negative_binomial(5, 0.3, (n_spots, n_genes)).astype(float)
    )
    spatial_adata.var_names = ref_adata.var_names
    
    # Create grid coordinates
    grid_size = int(np.ceil(np.sqrt(n_spots)))
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.flatten(), yy.flatten()])[:n_spots]
    spatial_adata.obsm['spatial'] = coords
    
    return ref_adata, spatial_adata


def run_dot_example(ref_adata, spatial_adata, device='cuda'):
    """
    Run DOT deconvolution on example data.
    
    Parameters
    ----------
    ref_adata : AnnData
        Reference single-cell data
    spatial_adata : AnnData
        Spatial transcriptomics data
    device : str
        'cuda' or 'cpu'
    """
    print("\n" + "="*60)
    print("Running DOT deconvolution")
    print("="*60)
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        print(f"Using device: {device}")
    
    # Step 1: Setup reference data
    print("\n1. Processing reference data...")
    ref_processed = setup_reference(
        ref_adata,
        cell_type_key='cell_type',
        subcluster_size=3,
        max_genes=300,
        verbose=True,
        device=device
    )
    
    # Step 2: Setup spatial data
    print("\n2. Processing spatial data...")
    spatial_processed = setup_spatial(
        spatial_adata,
        spatial_key='spatial',
        th_spatial=0.8,
        verbose=True,
        device=device
    )
    
    # Step 3: Create DOT object
    print("\n3. Creating DOT object...")
    dot = DOT(spatial_processed, ref_processed, ls_solution=True)
    
    # Step 4: Run optimization
    print("\n4. Running optimization...")
    dot.fit(
        mode='highres',  # or 'lowres' for low-resolution data
        ratios_weight=0.0,
        iterations=50,
        verbose=True
    )
    
    # Step 5: Get results
    print("\n5. Extracting results...")
    weights = dot.get_weights(normalize=True)
    cell_types = dot.get_cell_types()
    
    print(f"\nDeconvolution complete!")
    print(f"Weights shape: {weights.shape}")
    print(f"Cell types: {cell_types}")
    print(f"\nMean proportions:")
    for i, ct in enumerate(cell_types):
        print(f"  {ct}: {weights[:, i].mean():.3f}")
    
    return dot, weights, cell_types


def visualize_results(spatial_adata, dot, weights, cell_types):
    """Visualize DOT results."""
    print("\n" + "="*60)
    print("Visualizing results")
    print("="*60)
    
    coords = spatial_adata.obsm['spatial']
    
    # Plot spatial weights
    print("\nPlotting spatial distribution...")
    fig1 = plot_spatial_weights(
        coords,
        weights,
        cell_types=cell_types,
        ncols=3,
        point_size=20,
        title="Cell Type Spatial Distribution"
    )
    
    # Plot optimization history
    print("Plotting optimization history...")
    from dotr_pytorch.visualization import plot_optimization_history
    fig2 = plot_optimization_history(dot.history)
    
    # Plot proportions
    print("Plotting cell type proportions...")
    from dotr_pytorch.visualization import plot_cell_type_proportions
    fig3 = plot_cell_type_proportions(weights, cell_types=cell_types)
    
    print("\nVisualization complete!")
    
    return fig1, fig2, fig3


def main():
    """Main example workflow."""
    print("DOTpy Example")
    print("="*60)
    
    # Generate or load data
    ref_adata, spatial_adata = example_synthetic_data()
    
    # Run DOT
    dot, weights, cell_types = run_dot_example(
        ref_adata, 
        spatial_adata,
        device='cuda'  # Change to 'cpu' if no GPU
    )
    
    # Visualize
    visualize_results(spatial_adata, dot, weights, cell_types)
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


def example_with_real_data():
    """
    Example workflow with real spatial transcriptomics data.
    
    This assumes you have AnnData objects prepared with:
    - ref_adata: scRNA-seq with cell type annotations
    - spatial_adata: Spatial data with coordinates
    """
    # Load your data
    # ref_adata = sc.read_h5ad('path/to/reference.h5ad')
    # spatial_adata = sc.read_h5ad('path/to/spatial.h5ad')
    
    # Setup
    ref_processed = setup_reference(
        ref_adata,
        cell_type_key='cell_type',  # Your cell type annotation column
        subcluster_size=10,
        max_genes=5000,
        verbose=True
    )
    
    spatial_processed = setup_spatial(
        spatial_adata,
        spatial_key='spatial',  # Your spatial coordinates key
        th_spatial=0.84,
        verbose=True
    )
    
    # Create DOT object
    dot = DOT(spatial_processed, ref_processed)
    
    # Run for high-resolution data (e.g., Xenium, MERFISH, CosMx)
    dot.fit(
        mode='highres',
        ratios_weight=0.0,
        iterations=100,
        verbose=True
    )
    
    # Or for low-resolution data (e.g., Visium, ST)
    # dot.fit(
    #     mode='lowres',
    #     max_spot_size=20,
    #     ratios_weight=0.3,
    #     iterations=100,
    #     verbose=True
    # )
    
    # Get results
    weights = dot.get_weights(normalize=True)
    
    # Add to spatial AnnData
    spatial_adata.obsm['dot_weights'] = weights
    for i, ct in enumerate(dot.get_cell_types()):
        spatial_adata.obs[f'dot_{ct}'] = weights[:, i]
    
    # Visualize
    plot_spatial_weights(
        spatial_adata.obsm['spatial'],
        weights,
        cell_types=dot.get_cell_types()
    )
    
    return dot, weights


if __name__ == '__main__':
    # Run example
    main()
    
    # Keep plots open
    import matplotlib.pyplot as plt
    plt.show()
