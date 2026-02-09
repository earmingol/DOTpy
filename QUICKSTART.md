# DOTpy Quick Start Guide

## Installation

### Step 1: Install PyTorch

First, install PyTorch with CUDA support (for GPU acceleration):

```bash
# For CUDA 11.8 (check your CUDA version with: nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 2: Install DOTpy

```bash
cd DOTpy
pip install -e .
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

## Quick Test

Run the example script to verify installation:

```bash
cd DOTpy
python example.py
```

This will:
1. Generate synthetic data
2. Run DOT deconvolution
3. Create visualization plots

## First Real Analysis

### Prepare Your Data

Your data should be in AnnData format with:

**Reference scRNA-seq:**
- `adata.X`: Gene expression (cells × genes)
- `adata.obs['cell_type']`: Cell type annotations
- `adata.var_names`: Gene names

**Spatial transcriptomics:**
- `adata.X`: Gene expression (spots × genes)
- `adata.obsm['spatial']`: Spatial coordinates (spots × 2)
- `adata.var_names`: Gene names

### Basic Analysis

```python
import scanpy as sc
import torch
from dotpy import setup_reference, setup_spatial, DOT, plot_spatial_weights

# 1. Load your data
ref_adata = sc.read_h5ad('your_reference.h5ad')
spatial_adata = sc.read_h5ad('your_spatial.h5ad')

# 2. Process data
print("Processing reference...")
ref_processed = setup_reference(
    ref_adata,
    cell_type_key='cell_type',  # Your annotation column name
    subcluster_size=10,
    max_genes=5000,
    verbose=True
)

print("Processing spatial...")
spatial_processed = setup_spatial(
    spatial_adata,
    spatial_key='spatial',  # Your coordinate key
    th_spatial=0.84,
    verbose=True
)

# 3. Setup DOT with device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

dot = DOT(
    spatial_processed, 
    ref_processed,
    batch_size=500,  # Adjust based on GPU memory
    device=device
)

# 4. Run deconvolution
print("Running DOT...")
# For high-resolution data (Xenium, MERFISH, CosMx)
dot.fit(mode='highres', iterations=100, verbose=True)

# OR for low-resolution data (Visium, ST)
# dot.fit(mode='lowres', max_spot_size=20, iterations=100, verbose=True)

# 5. Get results
weights = dot.get_weights(normalize=True)
cell_types = dot.get_cell_types()

print(f"\nDeconvolution complete!")
print(f"Identified {len(cell_types)} cell types in {weights.shape[0]} spots")

# 6. Visualize
plot_spatial_weights(
    spatial_adata.obsm['spatial'],
    weights,
    cell_types=cell_types,
    ncols=4,
    save_path='cell_type_maps.png'
)

# 7. Save results
spatial_adata.obsm['dot_weights'] = weights
for i, ct in enumerate(cell_types):
    spatial_adata.obs[f'dot_{ct}'] = weights[:, i]

spatial_adata.write('spatial_deconvolved.h5ad')
print("Results saved!")
```

## Understanding the Parameters

### Reference Processing

```python
ref_processed = setup_reference(
    adata,
    cell_type_key='cell_type',  # Column with cell type labels
    subcluster_size=10,         # Max subclusters per cell type (higher = more granular)
    max_genes=5000,             # Number of genes to use (higher = more info, slower)
    remove_mt=True,             # Remove mitochondrial genes
    th_inner_logfold=0.75,      # Log-fold threshold for gene selection in subclustering
    random_state=42,            # Random seed for reproducibility
    verbose=True,               # Print progress
    copy=True                   # Copy adata before processing
)
```

**When to adjust:**
- Increase `subcluster_size` for more heterogeneous cell types
- Increase `max_genes` if you have many similar cell types
- Adjust `th_inner_logfold` to control gene selection stringency

### Spatial Processing

```python
spatial_processed = setup_spatial(
    adata,
    spatial_key='spatial',       # Key in adata.obsm with coordinates
    th_spatial=0.84,             # Similarity threshold for spatial neighbors
    th_gene_low=0.01,            # Min expression frequency
    th_gene_high=0.99,           # Max expression frequency
    remove_mt=True,              # Remove mitochondrial genes
    radius='auto',               # Spatial neighborhood radius
    verbose=True,
    copy=True
)
```

**When to adjust:**
- Lower `th_spatial` to include more neighbors (more smoothing)
- Set specific `radius` value if auto-detection fails
- Adjust `th_gene_low`/`th_gene_high` to filter genes by expression frequency

### DOT Initialization

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dot = DOT(
    spatial_processed,
    ref_processed,
    ls_solution=True,            # Use least-squares initialization (recommended)
    batch_size=500,              # Batch size for GPU processing
    device=device                # 'cuda' for GPU, 'cpu' for CPU
)
```

**When to adjust:**
- Decrease `batch_size` if running out of GPU memory
- Set `device='cpu'` if GPU memory is insufficient
- Set `ls_solution=False` to skip LS initialization (faster but may converge slower)

### DOT Fitting

```python
# High-resolution (subcellular)
dot.fit(
    mode='highres',              # For Xenium, MERFISH, CosMx, etc.
    ratios_weight=0.0,           # Weight for matching reference abundances (0-1)
    iterations=100,              # Number of optimization iterations
    gap_threshold=0.01,          # Convergence threshold
    use_mixed_precision=False,   # Use float16 on GPU (saves memory)
    checkpoint_dir=None,         # Directory to save checkpoints
    checkpoint_freq=10,          # Save checkpoint every N iterations
    resume_from=None,            # Path to checkpoint to resume from
    verbose=True
)

# Low-resolution (spot-based)
dot.fit(
    mode='lowres',               # For Visium, ST, etc.
    ratios_weight=0.3,           # Higher weight to match reference proportions
    max_spot_size=20,            # Max cells per spot
    iterations=100,
    gap_threshold=0.01,
    verbose=True
)
```

**When to adjust:**
- Increase `ratios_weight` if you trust reference proportions
- Increase `iterations` if not converging (check `dot.history`)
- Decrease `gap_threshold` for tighter convergence (slower)
- Enable `use_mixed_precision=True` on GPU to reduce memory usage

## Common Issues and Solutions

### Issue: "No common genes found"

**Solution:**
```python
# Check gene names
print("Reference genes (first 10):", ref_adata.var_names[:10])
print("Spatial genes (first 10):", spatial_adata.var_names[:10])

# Make sure they match (e.g., both upper/lowercase, same ID type)
ref_adata.var_names = ref_adata.var_names.str.upper()
spatial_adata.var_names = spatial_adata.var_names.str.upper()
```

### Issue: CUDA out of memory

**Solutions:**
```python
# Option 1: Reduce batch size
dot = DOT(spatial_processed, ref_processed, batch_size=100)

# Option 2: Use mixed precision
dot.fit(mode='highres', use_mixed_precision=True)

# Option 3: Reduce genes
ref_processed = setup_reference(
    ref_adata,
    max_genes=2000,  # Reduced from 5000
    ...
)

# Option 4: Use CPU
dot = DOT(spatial_processed, ref_processed, device='cpu')

# Option 5: Clear cache
import torch
torch.cuda.empty_cache()
```

### Issue: Slow convergence

**Solutions:**
```python
# Check optimization progress
from dotpy.visualization import plot_optimization_history
plot_optimization_history(dot.history)

# Try more iterations
dot.fit(iterations=200, verbose=True)

# Or looser convergence
dot.fit(gap_threshold=0.05, verbose=True)
```

### Issue: Results don't match expectations

**Checklist:**
1. Verify coordinate orientation (try `flip_y=True` in plots)
2. Check if gene names match exactly
3. Ensure cell type annotations are correct
4. Try different parameter values
5. Visualize reference cell types with UMAP

```python
# Check reference cell types
import matplotlib.pyplot as plt
sc.pp.neighbors(ref_adata)
sc.tl.umap(ref_adata)
sc.pl.umap(ref_adata, color='cell_type')
```

## Performance Tips

### For Large Datasets

```python
import torch

# 1. Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Reduce genes
max_genes = 3000  # Instead of 5000

# 3. Fewer subclusters
subcluster_size = 5  # Instead of 10

# 4. Use appropriate batch size
if device == 'cuda':
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    batch = 100 if mem_gb < 6 else (500 if mem_gb < 12 else 1000)
else:
    batch = 500

# 5. Subset spatial data for testing
spatial_subset = spatial_adata[::10].copy()  # Every 10th spot
```

### Monitor GPU Memory

```python
import torch

if torch.cuda.is_available():
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Clear cache if needed
    torch.cuda.empty_cache()
```

### Using Checkpoints

```python
# Save checkpoints during long runs
dot.fit(
    mode='highres',
    iterations=200,
    checkpoint_dir='./checkpoints',
    checkpoint_freq=20,  # Save every 20 iterations
    verbose=True
)

# Resume from checkpoint
dot.fit(
    mode='highres',
    iterations=300,  # Continue to 300 total
    resume_from='./checkpoints/checkpoint_iter_200.pkl',
    verbose=True
)
```

## Next Steps

1. **Validation**: Compare with known markers
   ```python
   # Visualize marker genes
   sc.pl.spatial(spatial_adata, color=['Gene_of_interest', 'dot_CellType'])
   ```

2. **Parameter tuning**: Try different settings
3. **Downstream analysis**: Use weights for further analysis
   ```python
   # Cluster based on cell type composition
   from sklearn.cluster import KMeans
   clusters = KMeans(n_clusters=5).fit_predict(weights)
   spatial_adata.obs['composition_cluster'] = clusters
   ```

4. **Publication-quality figures**: Adjust visualization parameters
   ```python
   plot_spatial_weights(
       spatial_adata.obsm['spatial'],
       weights,
       cell_types=cell_types,
       point_size=50,  # Larger points
       cmap='viridis',  # Different colormap
       figsize=(16, 12),  # Larger figure
       dpi=300,  # Publication quality
       save_path='figure.png'
   )
   ```

## Getting Help

- Check the full README.md for detailed documentation
- See R_TO_PYTHON_GUIDE.md for R comparison
- Run example.py for a complete workflow
- Check test_dotr.py for usage patterns

## Citation

If you use this software, please cite the original DOT paper and mention this PyTorch implementation.