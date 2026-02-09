"""
Example usage of DOTpy v0.2 – optimised implementation.

Demonstrates:
- Automatic CPU / GPU backend selection
- Mixed-precision optimisation
- Checkpointing & resuming
- Memory-constrained settings
"""

import numpy as np
import scanpy as sc
import torch
from pathlib import Path
from dotpy import DOT, setup_reference, setup_spatial, plot_spatial_weights


def check_gpu():
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  |  {props.total_memory / 1e9:.1f} GB")
        print(f"  allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        print("CUDA not available – using CPU")


# ---------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------

def _make_reference(n_cells=5000, n_genes=2000, n_types=5):
    """Generate synthetic reference single-cell data."""
    from scipy.sparse import random as sp_rand
    X = sp_rand(n_cells, n_genes, density=0.05, format='csr', dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, len(X.data)).astype(np.float32)
    adata = sc.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs['cell_type'] = np.random.choice(
        [f'CellType_{i}' for i in range(n_types)], size=n_cells
    )
    return adata


def _make_spatial(n_spots=1000, n_genes=2000):
    """Generate synthetic spatial transcriptomics data."""
    from scipy.sparse import random as sp_rand
    X = sp_rand(n_spots, n_genes, density=0.10, format='csr', dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, len(X.data)).astype(np.float32)
    adata = sc.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    g = int(np.ceil(np.sqrt(n_spots)))
    xx, yy = np.meshgrid(np.arange(g), np.arange(g))
    adata.obsm['spatial'] = np.column_stack([xx.ravel(), yy.ravel()])[:n_spots]
    return adata


# ---------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------

def example_workflow():
    """Main example workflow with standard settings."""
    print("=" * 70)
    print("DOTpy v0.2 – optimised workflow")
    print("=" * 70)
    check_gpu()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate synthetic data
    ref_adata = _make_reference()
    sp_adata = _make_spatial()
    print(f"\nReference: {ref_adata.shape}  |  Spatial: {sp_adata.shape}")

    # 1. Preprocessing
    print("\n-- Reference preprocessing --")
    ref = setup_reference(
        ref_adata,
        cell_type_key='cell_type',
        subcluster_size=10,
        max_genes=2000,
        verbose=True,
    )

    print("\n-- Spatial preprocessing --")
    sp = setup_spatial(
        sp_adata,
        spatial_key='spatial',
        th_spatial=0.80,
        verbose=True,
    )

    # 2. Determine batch size from available GPU memory
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch = 100 if mem_gb < 6 else (500 if mem_gb < 12 else 1000)
    else:
        batch = 500

    # 3. Run optimisation
    print(f"\n-- Optimisation (batch_size={batch}, device={device}) --")
    dot = DOT(sp, ref, ls_solution=True, batch_size=batch, device=device)

    ckpt_dir = './checkpoints_example'
    Path(ckpt_dir).mkdir(exist_ok=True)

    dot.fit(
        mode='highres',
        ratios_weight=0.0,
        iterations=30,
        gap_threshold=0.01,
        verbose=True,
        checkpoint_dir=ckpt_dir,
        checkpoint_freq=10,
        use_mixed_precision=(device == 'cuda'),
    )

    # 4. Results
    weights = dot.get_weights(normalize=True)
    cts = dot.get_cell_types()
    print(f"\nWeights: {weights.shape}  |  Cell types: {cts}")
    for i, ct in enumerate(cts):
        print(f"  {ct}: mean={weights[:, i].mean():.4f}")

    # 5. Visualize
    print("\n-- Creating visualizations --")
    from dotpy.visualization import plot_optimization_history

    plot_spatial_weights(
        sp_adata.obsm['spatial'],
        weights,
        cell_types=cts,
        ncols=3,
        save_path='cell_type_maps.png'
    )

    plot_optimization_history(
        dot.history,
        save_path='optimization_history.png'
    )

    return dot, weights, cts


def example_resume():
    """Resume from checkpoint."""
    print("\n" + "=" * 70)
    print("Resume from checkpoint")
    print("=" * 70)

    ckpt = './checkpoints_example/checkpoint_iter_20.pkl'
    if not Path(ckpt).exists():
        print(f"Checkpoint not found ({ckpt}). Run main example first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref = setup_reference(_make_reference(), cell_type_key='cell_type', verbose=False)
    sp = setup_spatial(_make_spatial(), verbose=False)

    dot = DOT(sp, ref, batch_size=500, device=device)

    print(f"Resuming optimization on {device}...")
    dot.fit(
        mode='highres',
        iterations=50,
        resume_from=ckpt,
        checkpoint_dir='./checkpoints_example',
        checkpoint_freq=10,
        verbose=True,
    )

    print("Resumed optimization complete!")


def example_memory_constrained():
    """Settings for ≤4 GB GPU or low-memory systems."""
    print("\n" + "=" * 70)
    print("Memory-constrained (4 GB GPU or CPU)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate smaller dataset
    ref_adata = _make_reference(n_cells=2000, n_genes=1500, n_types=4)
    sp_adata = _make_spatial(n_spots=500, n_genes=1500)

    print(f"Processing on {device} with memory-efficient settings...")

    ref = setup_reference(
        ref_adata,
        cell_type_key='cell_type',
        subcluster_size=5,       # Fewer subclusters
        max_genes=1000,          # Fewer genes
        verbose=True,
    )

    sp = setup_spatial(
        sp_adata,
        th_spatial=0.80,
        verbose=True
    )

    dot = DOT(sp, ref, batch_size=100, device=device)  # Small batch size

    dot.fit(
        mode='highres',
        iterations=20,
        verbose=True,
        use_mixed_precision=True,  # Use float16 to save memory
    )

    weights = dot.get_weights(normalize=True)
    print(f"Done (low-memory mode). Weights shape: {weights.shape}")


def example_high_quality():
    """High-quality settings for accurate results (slower)."""
    print("\n" + "=" * 70)
    print("High-quality mode (more iterations, tighter convergence)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref = setup_reference(
        _make_reference(),
        cell_type_key='cell_type',
        subcluster_size=15,      # More subclusters
        max_genes=5000,          # More genes
        verbose=True,
    )

    sp = setup_spatial(
        _make_spatial(),
        th_spatial=0.85,         # Stricter spatial threshold
        verbose=True
    )

    dot = DOT(sp, ref, batch_size=500, device=device)

    print(f"Running high-quality optimization on {device}...")
    dot.fit(
        mode='highres',
        iterations=200,          # More iterations
        gap_threshold=0.001,     # Tighter convergence
        verbose=True,
        checkpoint_dir='./checkpoints_hq',
        checkpoint_freq=25,
    )

    weights = dot.get_weights(normalize=True)
    print(f"Done (high-quality mode). Weights shape: {weights.shape}")


def example_visualization():
    """Demonstrate various visualization options."""
    print("\n" + "=" * 70)
    print("Visualization examples")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Quick run for visualization
    ref = setup_reference(_make_reference(), cell_type_key='cell_type', verbose=False)
    sp = setup_spatial(_make_spatial(), verbose=False)

    dot = DOT(sp, ref, device=device)
    dot.fit(mode='highres', iterations=30, verbose=False)

    weights = dot.get_weights(normalize=True)
    cts = dot.get_cell_types()
    coords = _make_spatial().obsm['spatial']

    # Different visualization styles
    from dotpy.visualization import plot_cell_type_proportions

    print("Creating various plots...")

    # Style 1: Default
    plot_spatial_weights(
        coords, weights, cell_types=cts,
        ncols=3, save_path='viz_default.png'
    )

    # Style 2: Custom colors and larger points
    plot_spatial_weights(
        coords, weights, cell_types=cts,
        ncols=3, point_size=20, cmap='viridis',
        save_path='viz_custom.png'
    )

    # Style 3: High-resolution
    plot_spatial_weights(
        coords, weights, cell_types=cts,
        ncols=2, figsize=(12, 10), dpi=300,
        save_path='viz_highres.png'
    )

    # Cell type proportions
    plot_cell_type_proportions(
        weights, cell_types=cts,
        save_path='cell_type_proportions.png'
    )

    print("All visualizations created!")


if __name__ == '__main__':
    # Run all examples
    print("\n" + "=" * 70)
    print("Running all DOTpy examples")
    print("=" * 70 + "\n")

    # Main workflow
    example_workflow()

    # Resume from checkpoint
    example_resume()

    # Memory-constrained
    example_memory_constrained()

    # High-quality
    example_high_quality()

    # Visualizations
    example_visualization()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)