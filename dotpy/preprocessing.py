"""
Preprocessing utilities for DOT algorithm

Memory-efficient implementation with optional GPU acceleration via rapids-singlecell.
Automatically selects scanpy (CPU) or rapids-singlecell (GPU) based on device parameter.
"""

import numpy as np
import torch
import scanpy as sc
from typing import Optional, Union, Tuple, Dict, List
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix, vstack
import warnings

# ---------------------------------------------------------------------------
# Backend helpers – transparently choose scanpy vs rapids_singlecell
# ---------------------------------------------------------------------------

_RSC_AVAILABLE: Optional[bool] = None


def _check_rapids():
    """Lazy-check whether rapids-singlecell is importable."""
    global _RSC_AVAILABLE
    if _RSC_AVAILABLE is None:
        try:
            import rapids_singlecell  # noqa: F401
            import cupy  # noqa: F401
            _RSC_AVAILABLE = True
        except ImportError:
            _RSC_AVAILABLE = False
    return _RSC_AVAILABLE


def _get_pp(device: str):
    """Return the preprocessing module (scanpy.pp or rapids_singlecell.pp)."""
    if device == 'cuda' and _check_rapids():
        import rapids_singlecell as rsc
        return rsc.pp
    return sc.pp


def _get_tl(device: str):
    """Return the tools module (scanpy.tl or rapids_singlecell.tl)."""
    if device == 'cuda' and _check_rapids():
        import rapids_singlecell as rsc
        return rsc.tl
    return sc.tl


def _to_gpu_anndata(adata: AnnData):
    """Move AnnData.X to GPU (cupy sparse) if rapids is available."""
    if _check_rapids():
        import rapids_singlecell as rsc
        rsc.get.anndata_to_GPU(adata)
    return adata


def _to_cpu_anndata(adata: AnnData):
    """Ensure AnnData.X is on CPU (scipy sparse / numpy)."""
    if _check_rapids():
        try:
            import rapids_singlecell as rsc
            rsc.get.anndata_to_CPU(adata)
        except Exception:
            pass
    return adata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_reference(
    adata: AnnData,
    cell_type_key: str,
    subcluster_size: int = 10,
    max_genes: int = 5000,
    remove_mt: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
    copy: bool = True
) -> Dict:
    """
    Process reference single-cell RNA-seq data for DOT.

    When ``device='cuda'`` and *rapids-singlecell* is installed the heavy
    lifting (normalisation, HVG selection, neighbour computation, Leiden
    clustering) runs on the GPU via cuML / cupy.  Otherwise scanpy is used
    transparently.

    Parameters
    ----------
    adata : AnnData
        Reference single-cell data with raw counts in .X
    cell_type_key : str
        Key in adata.obs containing cell type annotations
    subcluster_size : int
        Maximum number of sub-clusters per cell type
    max_genes : int
        Maximum number of genes to use
    remove_mt : bool
        Whether to remove mitochondrial / ribosomal / HLA genes
    verbose : bool
        Print progress messages
    device : str, optional
        ``'cuda'`` to attempt GPU preprocessing, ``'cpu'`` otherwise.
    copy : bool
        Whether to copy adata before processing

    Returns
    -------
    dict
        ``X_sparse``, ``clusters``, ``ratios``, ``genes``, ``device``
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_gpu_pp = device == 'cuda' and _check_rapids()
    pp = _get_pp(device)
    tl = _get_tl(device)

    if verbose:
        backend = "rapids-singlecell (GPU)" if use_gpu_pp else "scanpy (CPU)"
        print(f"Preprocessing reference data with {backend}...")
        print(f"Input shape: {adata.shape}")

    if copy:
        adata = adata.copy()

    # ==================================================================
    # Phase 1 – CPU-only operations (scanpy QC, gene filtering, counts)
    # Keep everything on CPU here; scanpy QC cannot handle cupy arrays.
    # ==================================================================
    if verbose:
        print("Running basic QC...")
    # sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=1)

    # Remove MT / HLA / RPL genes
    if remove_mt:
        mt_mask = adata.var_names.str.match('^MT-|^mt-|^RPL') # |^HLA-
        n_mt = mt_mask.sum()
        if n_mt > 0:
            adata = adata[:, ~mt_mask].copy()
            if verbose:
                print(f"Removed {n_mt} MT/RPL genes") # HLA/

    # Store raw counts layer (always CPU – needed later for aggregation)
    adata.layers['counts'] = adata.X.copy()

    # ==================================================================
    # Phase 2 – Normalisation & HVG (GPU-accelerated when available)
    # We snapshot the CPU counts *before* any GPU transfer because
    # anndata_to_GPU may convert layers to cupy as well.
    # ==================================================================
    counts_cpu = adata.X.copy()          # guaranteed scipy/numpy at this point
    all_gene_names = adata.var_names.tolist()

    if use_gpu_pp:
        _to_gpu_anndata(adata)

    if verbose:
        print("Normalising for HVG selection...")
    pp.normalize_total(adata, target_sum=1e4)
    pp.log1p(adata)

    # HVG selection – seurat_v3 needs CPU counts layer
    if adata.shape[1] > max_genes:
        if verbose:
            print(f"Selecting {max_genes} highly variable genes...")
        if use_gpu_pp:
            _to_cpu_anndata(adata)
        # Restore the original CPU counts for HVG ranking
        adata.layers['counts'] = counts_cpu
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=max_genes,
            flavor='seurat_v3',
            layer='counts',
            subset=False
        )
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        adata = adata[:, hvg_genes].copy()

    # --- Aggregation (always on CPU – small matrices) ---
    if use_gpu_pp:
        _to_cpu_anndata(adata)

    # Re-subset the CPU counts to match current (possibly HVG-filtered) genes
    current_genes = adata.var_names.tolist()
    if len(current_genes) != len(all_gene_names):
        gene_idx = np.array([all_gene_names.index(g) for g in current_genes])
        counts_cpu = counts_cpu[:, gene_idx]
    adata.layers['counts'] = counts_cpu

    X = adata.layers['counts']
    annotations = adata.obs[cell_type_key].values.astype(str)
    genes = adata.var_names.values

    if verbose:
        print(f"After filtering: {X.shape}")
        print("Aggregating and subclustering cell types...")

    ref_agg = _aggregate_reference(
        X, annotations, adata, subcluster_size,
        device=device, verbose=verbose
    )

    # --- DE gene selection ---
    if verbose:
        print("Selecting differentially expressed genes...")
    de_genes = _get_de_genes(
        ref_agg['sub_centroids'], ref_agg['clusters'], max_genes,
        verbose=verbose
    )

    X_subset = ref_agg['sub_centroids'][:, de_genes] if issparse(ref_agg['sub_centroids']) \
        else ref_agg['sub_centroids'][:, de_genes]

    result = {
        'X_sparse': X_subset,
        'clusters': ref_agg['clusters'],
        'ratios': ref_agg['major_ratios'],
        'genes': genes[de_genes],
        'device': device
    }

    if verbose:
        print(f"Reference prepared: {X_subset.shape[0]} subclusters, "
              f"{X_subset.shape[1]} genes")
        if issparse(X_subset):
            print(f"Sparsity: {1 - X_subset.nnz / (X_subset.shape[0] * X_subset.shape[1]):.2%}")

    return result


def setup_spatial(
    adata: AnnData,
    spatial_key: str = 'spatial',
    th_spatial: float = 0.84,
    th_nonspatial: float = 0.0,
    th_gene_low: float = 0.01,
    th_gene_high: float = 0.99,
    remove_mt: bool = True,
    n_neighbors: int = 8,
    radius: Union[str, float] = 'auto',
    verbose: bool = False,
    device: Optional[str] = None,
    copy: bool = True
) -> Dict:
    """
    Process spatial transcriptomics data for DOT.

    GPU-accelerated neighbour search via rapids-singlecell when available,
    otherwise uses sklearn ball-tree on CPU.

    Parameters
    ----------
    adata : AnnData
        Spatial data with raw counts in .X and coordinates in .obsm[spatial_key]
    spatial_key : str
        Key in adata.obsm for spatial coordinates
    th_spatial : float
        Cosine-similarity threshold for adjacent spots
    th_nonspatial : float
        Threshold for non-adjacent similar spots (0 = disabled)
    th_gene_low / th_gene_high : float
        Gene frequency filters
    remove_mt : bool
        Remove MT / HLA / RPL genes
    n_neighbors : int
        Neighbours used to estimate radius when ``radius='auto'``
    radius : float or ``'auto'``
        Spatial neighbourhood radius
    verbose : bool
        Print progress
    device : str, optional
        ``'cuda'`` or ``'cpu'``
    copy : bool
        Copy adata before processing

    Returns
    -------
    dict
        ``X_sparse``, ``coords``, ``pairs``, ``genes``, ``device``
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pp = _get_pp(device)
    use_gpu_pp = device == 'cuda' and _check_rapids()

    if verbose:
        backend = "rapids-singlecell (GPU)" if use_gpu_pp else "scanpy (CPU)"
        print(f"Preprocessing spatial data with {backend}...")
        print(f"Input shape: {adata.shape}")

    if copy:
        adata = adata.copy()

    # Coordinates
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.shape[1] > 2:
        coords = coords[:, :2]

    # Remove MT genes
    if remove_mt:
        mt_mask = adata.var_names.str.match('^MT-|^mt-|^RPL') # |^HLA-
        n_mt = mt_mask.sum()
        if n_mt > 0:
            adata = adata[:, ~mt_mask].copy()
            if verbose:
                print(f"Removed {n_mt} MT/RPL genes") # HLA/

    # Gene frequency filter
    sc.pp.filter_genes(adata, min_cells=int(th_gene_low * adata.shape[0]))
    gene_freq = np.asarray((adata.X > 0).mean(axis=0)).flatten()
    valid = gene_freq < th_gene_high
    adata = adata[:, valid].copy()

    if verbose:
        print(f"Filtered to {adata.shape[1]} genes")

    X = adata.X
    genes = adata.var_names.values

    result = {
        'X_sparse': X,
        'coords': coords,
        'genes': genes,
        'device': device
    }

    # Spatial pairs
    if th_spatial > 0:
        if verbose:
            print("Finding spatial neighbours...")
        pairs = _get_spatial_pairs(
            adata, coords, th_spatial, th_nonspatial,
            radius, n_neighbors, device, verbose
        )
        result['pairs'] = pairs

    if verbose:
        print(f"Spatial data prepared: {X.shape[0]} spots, {X.shape[1]} genes")
        if issparse(X):
            print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _make_pca_safe_layer(adata, eps=1e-6):
    X = adata.X
    if issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        var = X.var(axis=0)

    zero_var = var == 0
    if not np.any(zero_var):
        adata.layers["pca_safe"] = X.copy()
        return

    X_safe = X.copy()
    if issparse(X_safe):
        X_safe = X_safe.tolil()
        for g in np.where(zero_var)[0]:
            X_safe[np.random.randint(adata.n_obs), g] = eps
        X_safe = X_safe.tocsr()
    else:
        for g in np.where(zero_var)[0]:
            X_safe[np.random.randint(adata.n_obs), g] = eps

    adata.layers["pca_safe"] = X_safe


def _aggregate_reference(
    X, annotations, adata, cluster_size, device='cpu', verbose=False
):
    """Aggregate reference with subclustering (GPU-aware Leiden when possible)."""
    pp = _get_pp(device)
    tl = _get_tl(device)
    use_gpu = device == 'cuda' and _check_rapids()

    major_types = np.unique(annotations)
    major_centroids = []
    major_ratios = {}

    for ct in major_types:
        mask = annotations == ct
        if issparse(X):
            major_centroids.append(np.asarray(X[mask].mean(axis=0)).flatten())
        else:
            major_centroids.append(X[mask].mean(axis=0))
        major_ratios[ct] = int(mask.sum())

    total = sum(major_ratios.values())
    major_ratios = {k: v / total for k, v in major_ratios.items()}

    if issparse(X):
        major_centroids_mat = vstack([csr_matrix(c) for c in major_centroids])
    else:
        major_centroids_mat = np.array(major_centroids)

    if cluster_size <= 1:
        return {
            'major_centroids': major_centroids_mat,
            'major_ratios': major_ratios,
            'sub_centroids': major_centroids_mat,
            'clusters': {ct: [i] for i, ct in enumerate(major_types)}
        }

    # --- Subcluster each cell type ---
    sub_centroids = []
    clusters = {}

    for ct in major_types:
        mask = annotations == ct
        ct_adata = adata[mask].copy()

        if ct_adata.shape[0] <= 1:
            continue

        if ct_adata.shape[0] > 10000:
            sc.pp.subsample(ct_adata, n_obs=10000)

        _make_pca_safe_layer(ct_adata)

        K = min(cluster_size, max(1, int(2 * np.log(ct_adata.shape[0]) - 7)))

        if K <= 1:
            if issparse(ct_adata.layers['counts']):
                centroid = np.asarray(ct_adata.layers['counts'].mean(axis=0)).flatten()
            else:
                centroid = ct_adata.layers['counts'].mean(axis=0)
            sub_centroids.append(centroid)
            clusters[ct] = [len(sub_centroids) - 1]
        else:
            if verbose:
                print(f"  Clustering {ct_adata.shape[0]} {ct} cells into ~{K} clusters...")

            # Use rapids on GPU if available
            if use_gpu:
                _to_gpu_anndata(ct_adata)

            n_nbrs = min(15, ct_adata.shape[0] - 1)
            pp.pca(ct_adata, layer='pca_safe')
            pp.neighbors(ct_adata, n_neighbors=n_nbrs, use_rep='X_pca')
            resolution = K / 10.0
            tl.leiden(ct_adata, resolution=resolution, key_added='subcluster')

            if use_gpu:
                _to_cpu_anndata(ct_adata)

            sub_labels = ct_adata.obs['subcluster'].astype(str).values
            unique_labels = np.unique(sub_labels)

            cluster_ids = []
            # Ensure counts layer is on CPU (anndata_to_GPU may have converted it)
            counts_X = ct_adata.layers['counts']
            if hasattr(counts_X, 'get'):
                # cupy array/sparse → numpy/scipy
                counts_X = counts_X.get()
            for lab in unique_labels:
                lab_mask = sub_labels == lab
                if issparse(counts_X):
                    sc_centroid = np.asarray(counts_X[lab_mask].mean(axis=0)).flatten()
                else:
                    sc_centroid = counts_X[lab_mask].mean(axis=0)
                sub_centroids.append(sc_centroid)
                cluster_ids.append(len(sub_centroids) - 1)
            clusters[ct] = cluster_ids

    if issparse(X):
        sub_centroids_mat = vstack([csr_matrix(c) for c in sub_centroids])
    else:
        sub_centroids_mat = np.array(sub_centroids)

    return {
        'major_centroids': major_centroids_mat,
        'major_ratios': major_ratios,
        'sub_centroids': sub_centroids_mat,
        'clusters': clusters
    }


def _get_de_genes(centroids, clusters, max_genes, verbose=False):
    """Select DE genes using scanpy rank_genes_groups."""
    if centroids.shape[1] <= max_genes:
        return np.arange(centroids.shape[1])

    centroids_dense = centroids.toarray() if issparse(centroids) else centroids
    C, G = centroids_dense.shape

    adata_tmp = AnnData(X=centroids_dense)
    labels = []
    for ct, indices in clusters.items():
        labels.extend([ct] * len(indices))
    adata_tmp.obs['cell_type'] = labels

    # Only run if we have >1 group
    unique_cts = adata_tmp.obs['cell_type'].unique()
    if len(unique_cts) < 2:
        gene_var = centroids_dense.var(axis=0)
        return np.argsort(gene_var)[::-1][:max_genes]

    sc.tl.rank_genes_groups(adata_tmp, groupby='cell_type', method='wilcoxon', use_raw=False)

    de_set = set()
    n_per = max(50, max_genes // len(clusters))
    for ct in clusters:
        ct_genes = adata_tmp.uns['rank_genes_groups']['names'][ct][:n_per]
        de_set.update(ct_genes)

    if len(de_set) < max_genes:
        gene_var = centroids_dense.var(axis=0)
        for idx in np.argsort(gene_var)[::-1]:
            de_set.add(adata_tmp.var_names[idx])
            if len(de_set) >= max_genes:
                break

    de_idx = [adata_tmp.var_names.get_loc(g) for g in list(de_set)[:max_genes]]
    if verbose:
        print(f"Selected {len(de_idx)} DE genes")
    return np.array(de_idx)


def _get_spatial_pairs(
    adata, coords, th_spatial, th_nonspatial,
    radius, n_neighbors, device, verbose
):
    """
    Find spatial pairs.  Uses vectorised cosine similarity in batches
    to avoid O(N²) memory.  GPU-accelerated radius search via cuML
    when rapids is available.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize as sk_normalize

    N = coords.shape[0]

    # --- Estimate radius ---
    if radius == 'auto':
        nbrs_est = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, N),
            algorithm='ball_tree', metric='euclidean'
        )
        nbrs_est.fit(coords)
        dists, _ = nbrs_est.kneighbors(coords)
        radius = float(np.quantile(dists[:, -1], 0.9) * 1.05)
        if verbose:
            print(f"Estimated spatial radius: {radius:.2f}")

    # --- Find neighbours within radius ---
    if verbose:
        print(f"Finding spatial neighbours within radius {radius:.2f}...")

    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree', metric='euclidean')
    nbrs.fit(coords)
    distances_list, indices_list = nbrs.radius_neighbors(coords)

    # --- Normalise expression for cosine similarity ---
    X = adata.X
    if issparse(X):
        X_norm = sk_normalize(X, norm='l2', axis=1, copy=True)
    else:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms

    # --- Vectorised pair construction ---
    # Build flat arrays of (i, j) pairs from ball-tree results
    all_i = []
    all_j = []
    for i in range(N):
        nbr = indices_list[i]
        mask = nbr > i  # upper triangle only
        js = nbr[mask]
        if len(js) > 0:
            all_i.append(np.full(len(js), i, dtype=np.int64))
            all_j.append(js.astype(np.int64))

    if len(all_i) == 0:
        if verbose:
            print("No spatial pairs found within radius")
        return None

    i_idx = np.concatenate(all_i)
    j_idx = np.concatenate(all_j)

    # Compute cosine similarities in batches to limit memory
    PAIR_BATCH = 500_000
    n_pairs = len(i_idx)
    weights = np.empty(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, PAIR_BATCH):
        end = min(start + PAIR_BATCH, n_pairs)
        bi = i_idx[start:end]
        bj = j_idx[start:end]

        if issparse(X_norm):
            # Sparse row-wise dot product via element-wise multiply + sum
            Xi = X_norm[bi]
            Xj = X_norm[bj]
            sims = np.asarray(Xi.multiply(Xj).sum(axis=1)).flatten()
        else:
            sims = np.einsum('ij,ij->i', X_norm[bi], X_norm[bj])

        weights[start:end] = sims

    # Filter by threshold
    valid = weights >= th_spatial
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    weights = weights[valid]

    if verbose:
        print(f"Found {len(i_idx)} spatial pairs (th_spatial={th_spatial})")

    if len(i_idx) == 0:
        return None

    return {'i': i_idx, 'j': j_idx, 'w': weights}