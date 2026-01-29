# R to Python DOT Conversion Guide

This document provides a comparison between the original R implementation and the new PyTorch Python implementation.

## Quick Comparison

| Feature | R (DOTr) | Python (DOTpy) |
|---------|----------|----------------------|
| Main data structure | Seurat/Matrix | AnnData/PyTorch Tensor |
| Computation | CPU (R matrices) | GPU/CPU (PyTorch) |
| Dependencies | Seurat, Matrix, FNN | scanpy, PyTorch, scikit-learn |
| Speed | Baseline | ~10-100x faster on GPU |
| Memory efficiency | Good | Better (PyTorch optimization) |

## Function Mapping

### Data Setup

**R:**
```r
# Setup reference
dot.ref <- setup.ref(
  ref_data = seurat_obj,  # or matrix
  ref_annotations = "cell_type",
  ref_subcluster_size = 10,
  max_genes = 5000,
  remove_mt = TRUE,
  verbose = FALSE
)

# Setup spatial
dot.srt <- setup.srt(
  srt_data = seurat_obj,  # or matrix
  srt_coords = coords,
  th.spatial = 0.84,
  th.nonspatial = 0,
  remove_mt = TRUE,
  radius = 'auto',
  verbose = FALSE
)
```

**Python:**
```python
# Setup reference
ref_processed = setup_reference(
    adata=ref_adata,  # AnnData object
    cell_type_key='cell_type',
    subcluster_size=10,
    max_genes=5000,
    remove_mt=True,
    verbose=False,
    device='cuda'  # NEW: GPU support
)

# Setup spatial
spatial_processed = setup_spatial(
    adata=spatial_adata,  # AnnData object
    spatial_key='spatial',
    th_spatial=0.84,
    th_nonspatial=0.0,
    remove_mt=True,
    radius='auto',
    verbose=False,
    device='cuda'  # NEW: GPU support
)
```

### Creating DOT Object

**R:**
```r
dot <- create.DOT(dot.srt, dot.ref, ls_solution = TRUE)
```

**Python:**
```python
dot = DOT(spatial_processed, ref_processed, ls_solution=True)
```

### Running Optimization

**R:**
```r
# High resolution
dot <- run.DOT.highresolution(
  dot,
  ratios_weight = 0,
  iterations = 100,
  verbose = FALSE
)

# Low resolution
dot <- run.DOT.lowresolution(
  dot,
  ratios_weight = 0,
  max_spot_size = 20,
  iterations = 100,
  verbose = FALSE
)
```

**Python:**
```python
# High resolution
dot.fit(
    mode='highres',
    ratios_weight=0.0,
    iterations=100,
    verbose=False
)

# Low resolution
dot.fit(
    mode='lowres',
    ratios_weight=0.0,
    max_spot_size=20,
    iterations=100,
    verbose=False
)
```

### Accessing Results

**R:**
```r
# Get weights matrix
weights <- dot@weights

# Get solution
solution <- dot@solution

# Get history
history <- dot@history
```

**Python:**
```python
# Get weights as numpy array
weights = dot.get_weights(normalize=True)

# Get solution (PyTorch tensor)
solution = dot.solution

# Get history (dict)
history = dot.history
```

### Visualization

**R:**
```r
draw_maps(
  spatial = dot.srt$C,
  weights = dot@weights,
  normalize = TRUE,
  ncol = 4
)
```

**Python:**
```python
plot_spatial_weights(
    coords=spatial_adata.obsm['spatial'],
    weights=weights,
    cell_types=dot.get_cell_types(),
    ncols=4
)
```

## Algorithm Implementation Differences

### Core Optimization

Both implementations use the same Frank-Wolfe algorithm with identical objectives:

1. **Gene expression matching** (cosine distance)
2. **Spatial coherence** (Jensen-Shannon divergence)
3. **Cell type abundance matching** (JS divergence)
4. **Sparsity constraints**

**Key differences:**

| Component | R | Python |
|-----------|---|--------|
| Matrix operations | Base R matrices | PyTorch tensors |
| Normalization | Custom functions | `F.normalize()` |
| Distance computation | Manual loops | Vectorized GPU ops |
| Memory management | R garbage collection | PyTorch autograd + GC |

### Performance Optimizations in Python

1. **Vectorized operations** - All matrix operations use PyTorch
2. **GPU acceleration** - Automatic CUDA support
3. **Efficient memory** - PyTorch's memory pooling
4. **JIT compilation** - PyTorch's optimized kernels

## Data Format Conversion

### From Seurat to AnnData

```python
import scanpy as sc
from scipy.sparse import csr_matrix

# If you have Seurat RDS file
# 1. In R, save as h5Seurat
library(Seurat)
library(SeuratDisk)
SaveH5Seurat(seurat_obj, "seurat.h5Seurat")
Convert("seurat.h5Seurat", dest = "h5ad")

# 2. In Python, load
adata = sc.read_h5ad("seurat.h5ad")

# Or convert manually
# Export from R:
# counts <- GetAssayData(seurat_obj, slot = "counts")
# write.csv(as.matrix(counts), "counts.csv")
# write.csv(seurat_obj@meta.data, "metadata.csv")
# write.csv(GetTissueCoordinates(seurat_obj), "coords.csv")

# Import in Python:
import pandas as pd
counts = pd.read_csv("counts.csv", index_col=0)
metadata = pd.read_csv("metadata.csv", index_col=0)
coords = pd.read_csv("coords.csv", index_col=0)

adata = sc.AnnData(X=counts.T.values)  # Transpose: genes × cells -> cells × genes
adata.obs = metadata
adata.obsm['spatial'] = coords.values
```

### To Seurat from AnnData

```r
library(Seurat)
library(SeuratDisk)

# Convert h5ad to Seurat
Convert("output.h5ad", dest = "h5seurat")
seurat_obj <- LoadH5Seurat("output.h5seurat")

# Or manually:
# Save from Python:
# adata.write_csvs("output_dir")

# Load in R:
# counts <- read.csv("output_dir/X.csv", row.names = 1)
# metadata <- read.csv("output_dir/obs.csv", row.names = 1)
# seurat_obj <- CreateSeuratObject(counts = t(counts), meta.data = metadata)
```

## Example Workflows

### Complete R Workflow

```r
library(DOT)

# Load data
data(dot.sample)

# Setup
dot.ref <- setup.ref(
  dot.sample$ref$counts,
  dot.sample$ref$labels,
  ref_subcluster_size = 10
)

dot.srt <- setup.srt(
  dot.sample$srt$counts,
  dot.sample$srt$coordinates
)

# Create and run
dot <- create.DOT(dot.srt, dot.ref)
dot <- run.DOT.highresolution(dot, iterations = 100)

# Visualize
draw_maps(
  spatial = dot.srt$C,
  weights = dot@weights
)
```

### Complete Python Workflow

```python
import scanpy as sc
from dotpy import setup_reference, setup_spatial, DOT, plot_spatial_weights

# Load data
ref_adata = sc.read_h5ad('reference.h5ad')
spatial_adata = sc.read_h5ad('spatial.h5ad')

# Setup
ref_processed = setup_reference(
    ref_adata,
    cell_type_key='cell_type',
    subcluster_size=10
)

spatial_processed = setup_spatial(
    spatial_adata,
    spatial_key='spatial'
)

# Create and run
dot = DOT(spatial_processed, ref_processed)
dot.fit(mode='highres', iterations=100)

# Visualize
weights = dot.get_weights(normalize=True)
plot_spatial_weights(
    spatial_adata.obsm['spatial'],
    weights,
    cell_types=dot.get_cell_types()
)
```

## Migration Checklist

- [ ] Convert Seurat objects to AnnData
- [ ] Ensure gene names match between reference and spatial
- [ ] Check coordinate system orientation (may need to flip Y)
- [ ] Install PyTorch with CUDA support for GPU acceleration
- [ ] Test on small subset first
- [ ] Adjust parameters if needed (typically same as R)
- [ ] Validate results match R implementation

## Tips for Best Results

1. **Use same gene selection** - Use the same max_genes as R for comparison
2. **Match parameters** - Start with same parameters as R version
3. **Check convergence** - Both should converge to similar objectives
4. **GPU memory** - Monitor with `torch.cuda.memory_allocated()`
5. **Reproducibility** - Set random seeds if using subclustering

## Troubleshooting

### Different results from R?

1. Check gene overlap: `len(np.intersect1d(ref_genes, spatial_genes))`
2. Verify cell type annotations are loaded correctly
3. Ensure coordinates are in the same orientation
4. Check if random subclustering is affecting results (set seed)

### Out of memory errors?

1. Reduce `max_genes` parameter
2. Reduce `subcluster_size` parameter
3. Use CPU instead of GPU for very large datasets
4. Process spatial data in tiles

### Slow performance?

1. Ensure PyTorch is using GPU: `torch.cuda.is_available()`
2. Check if CPU-only PyTorch is installed
3. Reduce `iterations` for testing
4. Use fewer genes initially
