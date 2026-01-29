"""
Example usage of improved DOTpy implementation

Demonstrates memory-efficient processing with sparse matrices,
batched optimization, and checkpointing.
"""

import numpy as np
import scanpy as sc
import torch
from dotpy import DOT, setup_reference, setup_spatial
from pathlib import Path


def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {total_memory:.2f} GB")
        print(f"Current Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print("CUDA not available, using CPU")


def example_memory_efficient_workflow():
    """
    Complete memory-efficient workflow with all improvements.
    """
    print("="*70)
    print("Memory-Efficient DOTpy Example")
    print("="*70)
    
    # Check GPU
    check_gpu_memory()
    
    # Load your data
    print("\n1. Loading data...")
    # ref_adata = sc.read_h5ad('reference.h5ad')
    # spatial_adata = sc.read_h5ad('spatial.h5ad')
    
    # For demo, create synthetic data
    ref_adata = create_synthetic_reference()
    spatial_adata = create_synthetic_spatial()
    
    print(f"Reference: {ref_adata.shape}")
    print(f"Spatial: {spatial_adata.shape}")
    
    # Process reference - KEEPS SPARSE
    print("\n2. Processing reference (keeping sparse)...")
    ref_processed = setup_reference(
        ref_adata,
        cell_type_key='cell_type',
        subcluster_size=10,
        max_genes=2000,
        remove_mt=True,
        copy=True,  # Don't modify original
        verbose=True
    )
    
    print(f"\nReference processed:")
    print(f"  Subclusters: {ref_processed['X_sparse'].shape[0]}")
    print(f"  Genes: {ref_processed['X_sparse'].shape[1]}")
    if hasattr(ref_processed['X_sparse'], 'nnz'):
        sparsity = 1 - ref_processed['X_sparse'].nnz / (
            ref_processed['X_sparse'].shape[0] * ref_processed['X_sparse'].shape[1]
        )
        print(f"  Sparsity: {sparsity:.2%}")
    
    # Process spatial - KEEPS SPARSE
    print("\n3. Processing spatial (keeping sparse, using scanpy neighbors)...")
    spatial_processed = setup_spatial(
        spatial_adata,
        spatial_key='spatial',
        n_neighbors=8,
        th_spatial=0.80,
        verbose=True
    )
    
    print(f"\nSpatial processed:")
    print(f"  Spots: {spatial_processed['X_sparse'].shape[0]}")
    print(f"  Genes: {spatial_processed['X_sparse'].shape[1]}")
    if hasattr(spatial_processed['X_sparse'], 'nnz'):
        sparsity = 1 - spatial_processed['X_sparse'].nnz / (
            spatial_processed['X_sparse'].shape[0] * spatial_processed['X_sparse'].shape[1]
        )
        print(f"  Sparsity: {sparsity:.2%}")
    
    # Determine batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 6:
            batch_size = 100
        elif gpu_memory < 12:
            batch_size = 500
        else:
            batch_size = 1000
    else:
        batch_size = 500
    
    print(f"\n4. Creating DOT object with batch_size={batch_size}...")
    dot = DOT(
        spatial_processed,
        ref_processed,
        ls_solution=True,
        batch_size=batch_size
    )
    
    check_gpu_memory()
    
    # Run with checkpointing
    print("\n5. Running optimization with checkpointing...")
    checkpoint_dir = './checkpoints_example'
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    dot.fit(
        mode='highres',
        ratios_weight=0.0,
        iterations=30,  # Fewer for demo
        gap_threshold=0.01,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=10
    )
    
    check_gpu_memory()
    
    # Get results
    print("\n6. Extracting results...")
    weights = dot.get_weights(normalize=True)
    cell_types = dot.get_cell_types()
    
    print(f"\nResults:")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Cell types: {cell_types}")
    print(f"\nMean proportions:")
    for i, ct in enumerate(cell_types):
        print(f"    {ct}: {weights[:, i].mean():.4f}")
    
    return dot, weights, cell_types


def example_resume_from_checkpoint():
    """
    Example of resuming from a checkpoint after interruption.
    """
    print("\n" + "="*70)
    print("Resuming from Checkpoint Example")
    print("="*70)
    
    # Load data
    ref_adata = create_synthetic_reference()
    spatial_adata = create_synthetic_spatial()
    
    # Process
    ref_processed = setup_reference(
        ref_adata, cell_type_key='cell_type', verbose=False
    )
    spatial_processed = setup_spatial(spatial_adata, verbose=False)
    
    # Create DOT
    dot = DOT(spatial_processed, ref_processed, batch_size=500)
    
    # Resume from checkpoint
    checkpoint_path = './checkpoints_example/checkpoint_iter_20.pkl'
    
    if Path(checkpoint_path).exists():
        print(f"\nResuming from: {checkpoint_path}")
        
        dot.fit(
            mode='highres',
            iterations=50,
            resume_from=checkpoint_path,
            checkpoint_dir='./checkpoints_example',
            checkpoint_freq=10,
            verbose=True
        )
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run the main example first to create checkpoints")


def example_memory_constrained():
    """
    Example for very limited GPU memory (e.g., 4GB).
    """
    print("\n" + "="*70)
    print("Memory-Constrained Settings Example (4GB GPU)")
    print("="*70)
    
    ref_adata = create_synthetic_reference()
    spatial_adata = create_synthetic_spatial()
    
    # Aggressive memory reduction
    print("\n1. Aggressive preprocessing for low memory...")
    ref_processed = setup_reference(
        ref_adata,
        cell_type_key='cell_type',
        subcluster_size=5,  # Fewer subclusters
        max_genes=1000,  # Fewer genes
        verbose=True
    )
    
    spatial_processed = setup_spatial(
        spatial_adata,
        n_neighbors=6,  # Fewer neighbors
        verbose=True
    )
    
    # Very small batches
    print("\n2. Using small batch size (100)...")
    dot = DOT(
        spatial_processed,
        ref_processed,
        batch_size=100  # Small batches
    )
    
    dot.fit(
        mode='highres',
        iterations=20,
        verbose=True
    )
    
    print("\nOptimization completed on limited memory!")


def create_synthetic_reference():
    """Create synthetic reference scRNA-seq data."""
    n_cells = 5000
    n_genes = 2000
    n_cell_types = 5
    
    # Create sparse count matrix
    from scipy.sparse import random
    X = random(n_cells, n_genes, density=0.05, format='csr', dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, len(X.data)).astype(np.float32)
    
    adata = sc.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs['cell_type'] = np.random.choice(
        [f'CellType_{i}' for i in range(n_cell_types)],
        size=n_cells
    )
    
    return adata


def create_synthetic_spatial():
    """Create synthetic spatial transcriptomics data."""
    n_spots = 1000
    n_genes = 2000
    
    # Create sparse count matrix
    from scipy.sparse import random
    X = random(n_spots, n_genes, density=0.10, format='csr', dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, len(X.data)).astype(np.float32)
    
    adata = sc.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Grid coordinates
    grid_size = int(np.ceil(np.sqrt(n_spots)))
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.flatten(), yy.flatten()])[:n_spots]
    adata.obsm['spatial'] = coords
    
    return adata


def compare_memory_usage():
    """
    Compare memory usage between dense and sparse approaches.
    """
    print("\n" + "="*70)
    print("Memory Usage Comparison")
    print("="*70)
    
    ref_adata = create_synthetic_reference()
    
    print(f"\nReference data: {ref_adata.shape}")
    print(f"Sparsity: {1 - ref_adata.X.nnz / (ref_adata.shape[0] * ref_adata.shape[1]):.2%}")
    
    # Sparse size
    sparse_size = (ref_adata.X.data.nbytes + 
                   ref_adata.X.indices.nbytes + 
                   ref_adata.X.indptr.nbytes) / 1e6
    
    # Dense size
    dense_size = (ref_adata.shape[0] * ref_adata.shape[1] * 4) / 1e6  # float32
    
    print(f"\nMemory usage:")
    print(f"  Sparse matrix: {sparse_size:.2f} MB")
    print(f"  Dense matrix: {dense_size:.2f} MB")
    print(f"  Reduction: {dense_size / sparse_size:.1f}x")


def main():
    """Run all examples."""
    
    # Main workflow
    dot, weights, cell_types = example_memory_efficient_workflow()
    
    # Memory comparison
    compare_memory_usage()
    
    # Resume example (if checkpoints exist)
    example_resume_from_checkpoint()
    
    # Memory-constrained example
    example_memory_constrained()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)


if __name__ == '__main__':
    main()
