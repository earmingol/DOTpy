# DOTpy

**Deconvolution by Optimal Transport for Spatial Transcriptomics**

A Python implementation of the DOT algorithm for transferring cell type annotations from single-cell RNA-seq reference data to spatial transcriptomics data using multi-objective optimization.

## Features

- 🚀 **GPU acceleration** using PyTorch for fast computation
- 🧬 **AnnData integration** - seamlessly works with scanpy workflows
- 🎯 **Multi-objective optimization** using Frank-Wolfe algorithm
- 📊 **High & low resolution support** - works with both subcellular (Xenium, MERFISH, CosMx) and spot-based (Visium, ST) technologies
- 🎨 **Built-in visualization** tools for spatial cell type mapping
- 💾 **Checkpointing** for long-running optimizations
- ⚡ **Mixed precision** support for memory-efficient GPU training

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
from dotpy import DOT, setup_reference, setup_spatial, plot_spatial_weights

# Load data
ref_adata = sc.read_h5ad('reference.h5ad')
spatial_adata = sc.read_h5ad('spatial.h5ad')

# Process reference and spatial data
ref_processed = setup_reference(
    ref_adata,
    cell_type_key='cell_type',
    subcluster_size=10,
    max_genes=5000,
    verbose=True
)

spatial_processed = setup_spatial(
    spatial_adata,
    spatial_key='spatial',
    th_spatial=0.84,
    verbose=True
)

# Run DOT with batching
dot = DOT(
    spatial_processed, 
    ref_processed,
    batch_size=500  # Adjust for your GPU memory
)

dot.fit(
    mode='highres',
    iterations=100,
    checkpoint_dir='./checkpoints',  # Save checkpoints
    checkpoint_freq=10,
    verbose=True
)

# Get results
weights = dot.get_weights(normalize=True)
cell_types = dot.get_cell_types()

# Visualize results
plot_spatial_weights(
    spatial_adata.obsm['spatial'],
    weights,
    cell_types=cell_types,
    ncols=4,
    save_path='cell_type_maps.png'
)
```

### Resume from Checkpoint

```python
dot.fit(
    mode='highres',
    iterations=100,
    resume_from='./checkpoints/checkpoint_iter_50.pkl',
    verbose=True
)
```

## Command-Line Interface

For production workflows or batch processing, DOTpy includes a CLI script (`run_dot_cli.py`) that runs the full pipeline from the terminal without writing any Python code.

### Basic usage

```bash
python run_dot_cli.py --ref reference.h5ad --spatial spatial.h5ad
```

### Multi-sample processing

When a single AnnData object contains multiple tissue sections or slides, the CLI can iterate through each sample automatically using `--sample-key`. Each sample is preprocessed, deconvolved, and saved independently, with GPU memory freed between runs:

```bash
python run_dot_cli.py \
    --ref reference.h5ad \
    --spatial spatial_multi_slide.h5ad \
    --sample-key slide_id \
    --save-combined \
    -v
```

This produces per-sample results (`weights.csv`, `annotations.csv`, plots) and optionally a combined output with `--save-combined`.

### Resolution modes

```bash
# High-resolution (Xenium, MERFISH, CosMx) — default
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --mode highres

# Low-resolution (Visium, ST)
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --mode lowres --ratios-weight 0.3
```

### GPU acceleration and memory options

```bash
# Automatic GPU detection (default)
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --device auto

# Force CPU
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --device cpu

# Mixed precision for large datasets on GPU
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --device cuda --mixed-precision
```

### Checkpointing for long runs

```bash
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad \
    --checkpoint-dir ./checkpoints --checkpoint-freq 10

# Resume from a checkpoint
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad \
    --resume-from ./checkpoints/sample_1/checkpoint_iter_50.pkl
```

### Lineage-level annotation

When the reference contains a higher-level grouping (e.g., lineage or broad class), the CLI can map cell types to that level automatically:

```bash
python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad \
    --cell-type-key cell_type --lineage-key lineage
```

### Full example

```bash
python run_dot_cli.py \
    --ref reference.h5ad \
    --spatial visium_slides.h5ad \
    --sample-key slide_id \
    --cell-type-key cell_subclass \
    --lineage-key cell_class \
    --mode lowres \
    --ratios-weight 0.3 \
    --max-genes 5000 \
    --subcluster-size 10 \
    --batch-size 5000 \
    --iterations 100 \
    --device auto \
    --output dot_results \
    --output-dir ./results \
    --save-combined \
    -v
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--ref` | *(required)* | Path to reference scRNA-seq h5ad file |
| `--spatial` | *(required)* | Path to spatial transcriptomics h5ad file |
| `--sample-key` | `None` | Column in `obs` to split spatial data by sample/slide |
| `--cell-type-key` | `cell_type` | Column in reference `obs` with cell type labels |
| `--lineage-key` | `None` | Column in reference `obs` for higher-level grouping |
| `--counts-layer` | `counts` | Layer in spatial data with raw counts (`"X"` to use `.X` directly) |
| `--ref-counts-layer` | `X` | Layer in reference data with raw counts (`"X"` to use `.X` directly) |
| `--mode` | `highres` | `highres` or `lowres` |
| `--ratios-weight` | `0.0` | Weight for matching reference cell-type abundances |
| `--max-genes` | `5000` | Maximum genes for reference preprocessing |
| `--subcluster-size` | `10` | Maximum subclusters per cell type |
| `--th-spatial` | `0.84` | Cosine similarity threshold for spatial pairs |
| `--batch-size` | `5000` | Batch size for GPU processing |
| `--iterations` | `100` | Maximum Frank-Wolfe iterations |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--mixed-precision` | off | Use float16 intermediates on GPU |
| `--checkpoint-dir` | `None` | Directory for checkpoints |
| `--checkpoint-freq` | `10` | Checkpoint every N iterations |
| `--resume-from` | `None` | Resume from a checkpoint file |
| `--output` | `dot_results` | Output file prefix |
| `--output-dir` | `.` | Output directory |
| `--save-combined` | off | Merge per-sample results into one file |
| `--no-plots` | off | Skip plot generation |
| `--no-h5ad` | off | Skip saving per-sample h5ad files |
| `-v`, `--verbose` | off | Print detailed progress |

### Output files

For each sample, the CLI produces:

```
results/
  dot_results_slide1_weights.csv         # Cell-type weights per spot (S x K)
  dot_results_slide1_annotations.csv     # Dominant cell type per spot
  dot_results_slide1.h5ad                # Full AnnData with results
  figures/
    dot_results_slide1_cell_types.png    # Spatial cell type map
    dot_results_slide1_weights.png       # Per-type weight heatmaps
    dot_results_slide1_convergence.png   # Optimization convergence plot
  dot_results_combined_weights.csv       # (with --save-combined)
  dot_results_combined.h5ad              # (with --save-combined)
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
    subcluster_size=15,      # More subclusters per cell type
    max_genes=10000,         # Use more genes
    remove_mt=True,          # Remove mitochondrial genes
    th_inner_logfold=0.75,   # Log-fold threshold for gene selection
    random_state=42,         # For reproducibility
    verbose=True
)

# Setup spatial with custom thresholds
spatial_processed = setup_spatial(
    spatial_adata,
    spatial_key='spatial',
    th_spatial=0.80,         # Adjust spatial similarity threshold
    th_gene_low=0.01,        # Minimum gene expression frequency
    th_gene_high=0.99,       # Maximum gene expression frequency
    radius='auto',           # Or specify numeric value
    remove_mt=True,          # Remove mitochondrial genes
    verbose=True
)

# DOT with custom device and optimization settings
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dot = DOT(
    spatial_processed,
    ref_processed,
    batch_size=500,          # Adjust for GPU memory
    device=device            # Explicitly set device
)

# Fine-tune optimization
dot.fit(
    mode='highres',
    ratios_weight=0.2,       # Weight for abundance matching
    iterations=200,          # More iterations
    gap_threshold=0.001,     # Tighter convergence
    use_mixed_precision=True,  # Use float16 on GPU
    verbose=True
)
```

### GPU/CPU Selection

```python
# Check if CUDA is available
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Pass device to DOT
dot = DOT(
    spatial_processed,
    ref_processed,
    device=device
)
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

# Use smaller batch size
dot = DOT(spatial, ref, batch_size=100)

# Enable mixed precision on GPU
dot.fit(
    mode='highres',
    use_mixed_precision=True,
    iterations=100
)
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
Rahimi, A., Vale-Silva, L.A., Fälth Savitski, M. et al. 
DOT: a flexible multi-objective optimization framework for transferring features across single-cell and spatial omics. 
Nat Commun 15, 4994 (2024). https://doi.org/10.1038/s41467-024-48868-z
```

## Troubleshooting

### "CUDA out of memory"
```python
# Solution 1: Reduce batch size
dot = DOT(spatial, ref, batch_size=100)

# Solution 2: Enable mixed precision
dot.fit(mode='highres', use_mixed_precision=True)

# Solution 3: Use CPU
dot = DOT(spatial, ref, device='cpu')
```

### "Too slow on CPU"
```python
# Solution: Reduce data size
ref = setup_reference(adata, max_genes=2000, subcluster_size=5)
```

### No common genes found

```python
# Check gene names
print(f"Ref genes: {ref_adata.var_names[:10]}")
print(f"Spatial genes: {spatial_adata.var_names[:10]}")

# Ensure gene names match (e.g., both use same gene ID system)
```

### "Convergence issues"
```python
# Solution: More iterations or looser threshold
dot.fit(iterations=200, gap_threshold=0.05)
```

## Contact

For questions and issues, please open an issue on GitHub.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Disclaimer
This library was written in Python using Claude Sonnet 4.5 and GPT-5.2 models.


## License

MIT License