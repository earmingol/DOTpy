"""
Example usage of DOTpy v0.2 – optimised implementation.

Demonstrates:
- Automatic CPU / GPU backend selection for preprocessing
- Optional rapids-singlecell GPU preprocessing
- Mixed-precision optimisation
- Checkpointing & resuming
- Memory-constrained settings
"""

import numpy as np
import scanpy as sc
import torch
from pathlib import Path
from dotpy import DOT, setup_reference, setup_spatial


def check_gpu():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  |  {props.total_mem / 1e9:.1f} GB")
        print(f"  allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        print("CUDA not available – using CPU")


# ---------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------

def _make_reference(n_cells=5000, n_genes=2000, n_types=5):
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
    print("=" * 70)
    print("DOTpy v0.2 – optimised workflow")
    print("=" * 70)
    check_gpu()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref_adata = _make_reference()
    sp_adata = _make_spatial()
    print(f"\nReference: {ref_adata.shape}  |  Spatial: {sp_adata.shape}")

    # 1. Preprocessing (GPU-accelerated if rapids-singlecell is installed)
    print("\n-- Reference preprocessing --")
    ref = setup_reference(
        ref_adata,
        cell_type_key='cell_type',
        subcluster_size=10,
        max_genes=2000,
        device=device,
        verbose=True,
    )

    print("\n-- Spatial preprocessing --")
    sp = setup_spatial(
        sp_adata,
        spatial_key='spatial',
        th_spatial=0.80,
        device=device,
        verbose=True,
    )

    # 2. Determine batch size from available GPU memory
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        batch = 100 if mem_gb < 6 else (500 if mem_gb < 12 else 1000)
    else:
        batch = 500

    # 3. Run optimisation
    print(f"\n-- Optimisation (batch_size={batch}) --")
    dot = DOT(sp, ref, ls_solution=True, batch_size=batch)

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

    ref = setup_reference(_make_reference(), cell_type_key='cell_type', verbose=False)
    sp = setup_spatial(_make_spatial(), verbose=False)
    dot = DOT(sp, ref, batch_size=500)

    dot.fit(
        mode='highres',
        iterations=50,
        resume_from=ckpt,
        checkpoint_dir='./checkpoints_example',
        checkpoint_freq=10,
        verbose=True,
    )


def example_memory_constrained():
    """Settings for ≤4 GB GPU."""
    print("\n" + "=" * 70)
    print("Memory-constrained (4 GB GPU)")
    print("=" * 70)

    ref = setup_reference(
        _make_reference(),
        cell_type_key='cell_type',
        subcluster_size=5,
        max_genes=1000,
        verbose=True,
    )
    sp = setup_spatial(_make_spatial(), n_neighbors=6, verbose=True)

    dot = DOT(sp, ref, batch_size=100)
    dot.fit(
        mode='highres',
        iterations=20,
        verbose=True,
        use_mixed_precision=True,
    )
    print("Done (low-memory mode).")


if __name__ == '__main__':
    example_workflow()
    example_resume()
    example_memory_constrained()
    print("\n" + "=" * 70 + "\nAll examples complete.\n" + "=" * 70)