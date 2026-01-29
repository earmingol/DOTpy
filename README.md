# DOTpy

**Deconvolution by Optimal Transport for Spatial Transcriptomics**

A PyTorch implementation of the DOT algorithm for transferring cell type annotations from single-cell RNA-seq reference data to spatial transcriptomics data using multi-objective optimization.

## Features

- 🚀 **GPU acceleration** using PyTorch for fast computation
- 🧬 **AnnData integration** - seamlessly works with scanpy workflows
- 🎯 **Multi-objective optimization** using Frank-Wolfe algorithm
- 📊 **High & low resolution support** - works with both subcellular (Xenium, MERFISH, CosMx) and spot-based (Visium, ST) technologies
- 🎨 **Built-in visualization** tools for spatial cell type mapping

## Installation

### From source

```bash
git clone https://github.com/earmingol/DOTpy.git
cd DOTpy
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0 (with CUDA support for GPU acceleration)
- scanpy >= 1.9.0
- anndata >= 0.8.0
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

## Quick Start

### Basic Usage

```python
import scanpy as sc
from dotpy import setup_reference, setup_spatial, DOT, plot_spatial_weights

# Load your data
ref_adata = sc.read_h5ad('reference_scrna.h5ad')
spatial_adata = sc.read_h5ad('spatial_data.h5ad')

# Process reference data
ref_processed = setup_reference(
    ref_adata,
    cell_type_key='cell_type',  # Column in ref_adata.obs with cell types
    subcluster_size=10,
    max_genes=5000,
    verbose=True
)

# Process spatial data
spatial_processed = setup_spatial(
    spatial_adata,
    spatial_key='spatial',  # Key in spatial_adata.obsm with coordinates
    th_spatial=0.84,
    verbose=True
)

# Create DOT object and run deconvolution
dot = DOT(spatial_processed, ref_processed)
dot.fit(mode='highres', iterations=100, verbose=True)

# Get cell type weights
weights = dot.get_weights(normalize=True)
cell_types = dot.get_cell_types()

# Visualize results
plot_spatial_weights(
    spatial_adata.obsm['spatial'],
    weights,
    cell_types=cell_types
)
```

### High-Resolution Data (Xenium, MERFISH, CosMx)

For subcellular resolution data where each spot typically contains 1 cell:

```python
dot.fit(
    mode='highres',
    ratios_weight=0.0,
    iterations=100,
    verbose=True
)
```

### Low-Resolution Data (Visium, ST)

For spot-based technologies where spots contain multiple cells:

```python
dot.fit(
    mode='lowres',
    max_spot_size=20,  # Maximum cells per spot
    ratios_weight=0.3,  # Weight for matching cell type proportions
    iterations=100,
    verbose=True
)
```

## Algorithm Overview

DOT uses multi-objective optimization to find cell type assignments that:

1. **Match gene expression** - Predicted expression should match observed spatial data
2. **Preserve spatial coherence** - Neighboring spots should have similar composition
3. **Respect cell type abundances** - Overall proportions should match reference (optional)
4. **Enforce sparsity** - Limit mixing of cell types per spot

The optimization is performed using the Frank-Wolfe algorithm, which efficiently handles the constrained optimization problem on GPUs.

## Advanced Usage

### Custom Parameters

```python
# Setup reference with custom parameters
ref_processed = setup_reference(
    ref_adata,
    cell_type_key='cell_type',
    subcluster_size=15,  # More subclusters per cell type
    max_genes=10000,      # Use more genes
    remove_mt=True,       # Remove mitochondrial genes
    verbose=True
)

# Setup spatial with custom thresholds
spatial_processed = setup_spatial(
    spatial_adata,
    spatial_key='spatial',
    th_spatial=0.80,       # Adjust spatial similarity threshold
    th_nonspatial=0.0,     # Include non-adjacent similar spots
    th_gene_low=0.01,      # Minimum gene expression frequency
    th_gene_high=0.99,     # Maximum gene expression frequency
    radius='auto',         # Or specify numeric value
    verbose=True
)

# Fine-tune optimization
dot.fit(
    mode='highres',
    ratios_weight=0.2,     # Weight for abundance matching
    iterations=200,         # More iterations
    gap_threshold=0.001,    # Tighter convergence
    verbose=True
)
```

### GPU/CPU Selection

```python
# Explicitly specify device
ref_processed = setup_reference(
    ref_adata,
    cell_type_key='cell_type',
    device='cuda'  # or 'cpu'
)

# Check if CUDA is available
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
```

### Saving Results

```python
# Add results to spatial AnnData
spatial_adata.obsm['dot_weights'] = weights

# Add individual cell type columns
for i, ct in enumerate(cell_types):
    spatial_adata.obs[f'dot_{ct}'] = weights[:, i]

# Save
spatial_adata.write('spatial_with_deconvolution.h5ad')
```

## Visualization

### Spatial Cell Type Maps

```python
from dotpy.visualization import plot_spatial_weights

fig = plot_spatial_weights(
    coords=spatial_adata.obsm['spatial'],
    weights=weights,
    cell_types=cell_types,
    ncols=4,
    point_size=10,
    cmap='magma',
    flip_y=True,
    save_path='cell_type_maps.png',
    dpi=300
)
```

### Optimization History

```python
from dotpy.visualization import plot_optimization_history

fig = plot_optimization_history(
    dot.history,
    save_path='optimization_history.png'
)
```

### Cell Type Proportions

```python
from dotpy.visualization import plot_cell_type_proportions

fig = plot_cell_type_proportions(
    weights,
    cell_types=cell_types,
    save_path='proportions.png'
)
```

## Performance Tips

### GPU Acceleration

DOTpy automatically uses CUDA if available. For best performance:

```python
# Check GPU memory
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Memory Management

For very large datasets:

```python
# Reduce number of genes
ref_processed = setup_reference(
    ref_adata,
    max_genes=2000,  # Use fewer genes
    ...
)

# Reduce subclustering
ref_processed = setup_reference(
    ref_adata,
    subcluster_size=5,  # Fewer subclusters
    ...
)

# Process in batches if needed
# (split spatial data into tiles)
```

### Speed vs Accuracy

```python
# Faster (fewer iterations)
dot.fit(mode='highres', iterations=50)

# More accurate (more iterations, tighter convergence)
dot.fit(
    mode='highres',
    iterations=200,
    gap_threshold=0.001
)
```

## Comparison with R Implementation

This PyTorch implementation provides:

- ✅ **Faster computation** through GPU acceleration
- ✅ **Same algorithm** and mathematical formulation
- ✅ **AnnData integration** for Python/scanpy workflows
- ✅ **Memory efficiency** through PyTorch's optimized operations

Key differences:
- Uses PyTorch tensors instead of R matrices
- Integrates with scanpy/AnnData instead of Seurat
- Supports GPU acceleration out of the box

## Citation

If you use DOT in your research, please cite:

```
[Original DOT paper citation]
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size or number of genes
ref_processed = setup_reference(ref_adata, max_genes=2000)
```

### No common genes found

```python
# Check gene names
print(f"Ref genes: {ref_adata.var_names[:10]}")
print(f"Spatial genes: {spatial_adata.var_names[:10]}")

# Ensure gene names match (e.g., both use same gene ID system)
```

### Slow convergence

```python
# Try adjusting parameters
dot.fit(
    mode='highres',
    iterations=150,
    gap_threshold=0.01  # Looser convergence
)
```

## Contact

For questions and issues, please open an issue on GitHub.
