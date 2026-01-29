"""
Preprocessing utilities for DOT algorithm

Memory-efficient implementation using scanpy built-in functions and sparse matrices.
"""

import numpy as np
import torch
import scanpy as sc
from typing import Optional, Union, Tuple, Dict, List
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix, vstack
import warnings


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
    
    Uses scanpy functions for efficiency and keeps matrices sparse.
    
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
        Whether to remove mitochondrial genes
    verbose : bool
        Print progress messages
    device : str, optional
        Device for PyTorch tensors ('cuda' or 'cpu')
    copy : bool
        Whether to copy adata before processing
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Gene expression centroids (subclusters × genes) as sparse matrix
        - 'clusters': Dictionary mapping cell types to subcluster indices
        - 'ratios': Cell type abundance ratios
        - 'genes': Gene names
        - 'device': Device being used
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("Preprocessing reference data with scanpy...")
        print(f"Using device: {device}")
        print(f"Input shape: {adata.shape}")
    
    # Copy to avoid modifying original
    if copy:
        adata = adata.copy()
    
    # Basic QC using scanpy
    if verbose:
        print("Running basic QC...")
    
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Filter cells with zero counts
    sc.pp.filter_cells(adata, min_counts=1)
    
    # Remove MT genes if requested
    if remove_mt:
        adata.var['mt'] = adata.var_names.str.match('^MT-|^mt-|^HLA-|^RPL')
        n_mt = adata.var['mt'].sum()
        if n_mt > 0:
            adata = adata[:, ~adata.var['mt']].copy()
            if verbose:
                print(f"Removed {n_mt} mitochondrial genes")
    
    # Normalize and log-transform for HVG selection
    if verbose:
        print("Normalizing for HVG selection...")
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes using scanpy
    if adata.shape[1] > max_genes:
        if verbose:
            print(f"Selecting {max_genes} highly variable genes...")
        
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=max_genes,
            flavor='seurat_v3',
            layer='counts',
            subset=False
        )
        
        # Subset to HVGs
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        adata = adata[:, hvg_genes].copy()
    
    # Use raw counts for aggregation
    X = adata.layers['counts']
    annotations = adata.obs[cell_type_key].values.astype(str)
    genes = adata.var_names.values
    
    if verbose:
        print(f"After filtering: {X.shape}")
        print("Aggregating and subclustering cell types...")
    
    # Aggregate with subclustering
    ref_agg = _aggregate_reference_scanpy(
        X,
        annotations,
        adata,  # Pass full adata for scanpy functions
        subcluster_size,
        verbose=verbose
    )
    
    # Select DE genes
    if verbose:
        print("Selecting differentially expressed genes...")
    
    de_genes = _get_de_genes_scanpy(
        ref_agg['sub_centroids'],
        ref_agg['clusters'],
        max_genes,
        verbose=verbose
    )
    
    # Subset to DE genes (keep sparse)
    if issparse(ref_agg['sub_centroids']):
        X_subset = ref_agg['sub_centroids'][:, de_genes]
    else:
        X_subset = ref_agg['sub_centroids'][:, de_genes]
    
    # Prepare output
    result = {
        'X_sparse': X_subset,  # Keep as sparse matrix
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

    Uses scanpy for finding spatial neighbors efficiently.

    Parameters
    ----------
    adata : AnnData
        Spatial data with raw counts in .X and coordinates in .obsm[spatial_key]
    spatial_key : str
        Key in adata.obsm containing spatial coordinates
    th_spatial : float
        Threshold on similarity of adjacent spots
    th_nonspatial : float
        Threshold on similarity of non-adjacent spots
    th_gene_low : float
        Minimum expression frequency for genes
    th_gene_high : float
        Maximum expression frequency for genes
    remove_mt : bool
        Whether to remove mitochondrial genes
    n_neighbors : int
        Number of spatial neighbors for estimating radius (if radius='auto')
    radius : float or 'auto'
        Spatial neighborhood radius. If 'auto', estimated from coordinates
    verbose : bool
        Print progress messages
    device : str, optional
        Device for PyTorch tensors
    copy : bool
        Whether to copy adata

    Returns
    -------
    dict
        Dictionary containing:
        - 'X_sparse': Gene expression matrix (spots × genes) as sparse
        - 'coords': Spatial coordinates
        - 'pairs': Spatial neighbor pairs (if th_spatial > 0)
        - 'genes': Gene names
        - 'device': Device being used
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print("Preprocessing spatial data with scanpy...")
        print(f"Using device: {device}")
        print(f"Input shape: {adata.shape}")

    if copy:
        adata = adata.copy()

    # Extract coordinates
    if spatial_key in adata.obsm:
        coords = np.array(adata.obsm[spatial_key])
    else:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")

    # Take first 2 columns if more
    if coords.shape[1] > 2:
        coords = coords[:, :2]

    # Remove MT genes
    if remove_mt:
        adata.var['mt'] = adata.var_names.str.match('^MT-|^mt-|^HLA-|^RPL')
        n_mt = adata.var['mt'].sum()
        if n_mt > 0:
            adata = adata[:, ~adata.var['mt']].copy()
            if verbose:
                print(f"Removed {n_mt} mitochondrial genes")

    # Filter genes by expression frequency
    sc.pp.filter_genes(adata, min_cells=int(th_gene_low * adata.shape[0]))

    gene_freq = (adata.X > 0).mean(axis=0)
    if issparse(adata.X):
        gene_freq = np.asarray(gene_freq).flatten()

    valid_genes = gene_freq < th_gene_high
    adata = adata[:, valid_genes].copy()

    if verbose:
        print(f"Filtered to {adata.shape[1]} genes")

    # Store counts
    X = adata.X
    genes = adata.var_names.values

    result = {
        'X_sparse': X,  # Keep sparse
        'coords': coords,
        'genes': genes,
        'device': device
    }

    # Find spatial pairs using scanpy's spatial neighbors
    if th_spatial > 0:
        if verbose:
            print(f"Finding spatial neighbors...")

        pairs = _get_spatial_pairs_scanpy(
            adata,
            result['coords'],
            th_spatial,
            th_nonspatial,
            radius,
            n_neighbors,
            verbose
        )

        result['pairs'] = pairs

    if verbose:
        print(f"Spatial data prepared: {X.shape[0]} spots, {X.shape[1]} genes")
        if issparse(X):
            print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")

    return result


def _aggregate_reference_scanpy(
    X: Union[np.ndarray, csr_matrix],
    annotations: np.ndarray,
    adata: AnnData,
    cluster_size: int,
    verbose: bool = False
) -> Dict:
    """
    Aggregate reference using scanpy's clustering.

    Uses Leiden clustering on normalized data per cell type.
    """
    major_types = np.unique(annotations)

    # Compute major type centroids and ratios
    major_centroids = []
    major_ratios = {}

    for ct in major_types:
        ct_mask = annotations == ct
        if issparse(X):
            ct_centroid = np.asarray(X[ct_mask].mean(axis=0)).flatten()
        else:
            ct_centroid = X[ct_mask].mean(axis=0)

        major_centroids.append(ct_centroid)
        major_ratios[ct] = ct_mask.sum()

    if issparse(X):
        major_centroids = vstack([csr_matrix(c) for c in major_centroids])
    else:
        major_centroids = np.array(major_centroids)

    total = sum(major_ratios.values())
    major_ratios = {k: v/total for k, v in major_ratios.items()}

    # Subcluster if requested
    if cluster_size > 1:
        if verbose:
            print("Subclustering cell types with Leiden...")

        sub_centroids = []
        clusters = {}

        for ct in major_types:
            ct_mask = annotations == ct
            ct_adata = adata[ct_mask].copy()

            if ct_adata.shape[0] <= 1:
                continue

            # Subsample if too many cells
            if ct_adata.shape[0] > 10000:
                sc.pp.subsample(ct_adata, n_obs=10000)

            # Determine number of clusters
            K = min(cluster_size, max(1, int(2 * np.log(ct_adata.shape[0]) - 7)))

            if K <= 1:
                # Just take mean
                if issparse(ct_adata.layers['counts']):
                    centroid = np.asarray(ct_adata.layers['counts'].mean(axis=0)).flatten()
                else:
                    centroid = ct_adata.layers['counts'].mean(axis=0)

                sub_centroids.append(centroid)
                clusters[ct] = [len(sub_centroids) - 1]
            else:
                # Use scanpy for clustering
                if verbose:
                    print(f"  Clustering {ct_adata.shape[0]} {ct} cells into {K} clusters...")

                # Compute neighbors
                sc.pp.neighbors(ct_adata, n_neighbors=min(15, ct_adata.shape[0] - 1))

                # Leiden clustering with target resolution
                resolution = K / 10.0  # Heuristic
                sc.tl.leiden(ct_adata, resolution=resolution, key_added='subcluster')

                # Get unique clusters
                sub_labels = ct_adata.obs['subcluster'].astype(str).values
                unique_clusters = np.unique(sub_labels)

                # Compute centroids for each subcluster
                cluster_ids = []
                for sc_label in unique_clusters:
                    sc_mask = sub_labels == sc_label

                    if issparse(ct_adata.layers['counts']):
                        sc_centroid = np.asarray(
                            ct_adata.layers['counts'][sc_mask].mean(axis=0)
                        ).flatten()
                    else:
                        sc_centroid = ct_adata.layers['counts'][sc_mask].mean(axis=0)

                    sub_centroids.append(sc_centroid)
                    cluster_ids.append(len(sub_centroids) - 1)

                clusters[ct] = cluster_ids

        if issparse(X):
            sub_centroids = vstack([csr_matrix(c) for c in sub_centroids])
        else:
            sub_centroids = np.array(sub_centroids)
    else:
        sub_centroids = major_centroids
        clusters = {ct: [i] for i, ct in enumerate(major_types)}

    return {
        'major_centroids': major_centroids,
        'major_ratios': major_ratios,
        'sub_centroids': sub_centroids,
        'clusters': clusters
    }


def _get_de_genes_scanpy(
    centroids: Union[np.ndarray, csr_matrix],
    clusters: Dict,
    max_genes: int,
    verbose: bool = False
) -> np.ndarray:
    """Select DE genes using scanpy's ranking."""
    if centroids.shape[1] <= max_genes:
        return np.arange(centroids.shape[1])

    # Convert to dense for DE analysis if sparse
    if issparse(centroids):
        centroids_dense = centroids.toarray()
    else:
        centroids_dense = centroids

    C, G = centroids_dense.shape

    # Create temporary AnnData for scanpy's rank_genes_groups
    adata_tmp = AnnData(X=centroids_dense)

    # Add cell type labels
    cell_types = []
    for ct, indices in clusters.items():
        cell_types.extend([ct] * len(indices))
    adata_tmp.obs['cell_type'] = cell_types

    # Rank genes
    sc.tl.rank_genes_groups(
        adata_tmp,
        groupby='cell_type',
        method='wilcoxon',
        use_raw=False
    )

    # Get top DE genes per cell type
    de_genes_set = set()
    n_per_type = max(50, max_genes // len(clusters))

    for ct in clusters.keys():
        ct_genes = adata_tmp.uns['rank_genes_groups']['names'][ct][:n_per_type]
        de_genes_set.update(ct_genes)

    # If not enough, add by variance
    if len(de_genes_set) < max_genes:
        gene_var = centroids_dense.var(axis=0)
        top_var = np.argsort(gene_var)[::-1]

        for idx in top_var:
            gene_name = adata_tmp.var_names[idx]
            de_genes_set.add(gene_name)
            if len(de_genes_set) >= max_genes:
                break

    # Convert to indices
    de_genes = [adata_tmp.var_names.get_loc(g) for g in list(de_genes_set)[:max_genes]]

    if verbose:
        print(f"Selected {len(de_genes)} DE genes")

    return np.array(de_genes)


def _get_spatial_pairs_scanpy(
    adata: AnnData,
    coords: np.ndarray,
    th_spatial: float,
    th_nonspatial: float,
    radius: Union[str, float],
    n_neighbors: int,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Find spatial pairs using ball tree for efficient radius-based neighbor finding.

    Much faster than pairwise distances: O(N log N) instead of O(N^2).

    Matches R implementation:
    1. Find all spots within radius (using ball tree)
    2. Compute transcriptomic similarity (cosine)
    3. Filter by similarity threshold
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize as sk_normalize

    N = coords.shape[0]

    # Compute radius if auto
    if radius == 'auto':
        # Use ball tree for k-NN to estimate radius
        nbrs_estimator = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, N),
            algorithm='ball_tree',
            metric='euclidean'
        )
        nbrs_estimator.fit(coords)
        distances, _ = nbrs_estimator.kneighbors(coords)

        # Get max distance to k-th neighbor (excluding self)
        max_dists = distances[:, -1]

        # Take 90th percentile and add 5% margin
        radius = float(np.quantile(max_dists, 0.9) * 1.05)

        if verbose:
            print(f"Estimated spatial radius: {radius:.2f}")

    # Normalize expression for similarity (L2 norm)
    X = adata.X
    if issparse(X):
        X_norm = sk_normalize(X, norm='l2', axis=1)
    else:
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    # Use ball tree to find ALL neighbors within radius (EFFICIENT!)
    if verbose:
        print(f"Finding spatial neighbors within radius {radius:.2f}...")

    nbrs = NearestNeighbors(
        radius=radius,
        algorithm='ball_tree',
        metric='euclidean'
    )
    nbrs.fit(coords)

    # Query all points for neighbors within radius
    # Returns: indices and distances for each point
    distances_list, indices_list = nbrs.radius_neighbors(coords)

    # Build pairs list
    all_i = []
    all_j = []
    all_weights = []

    for i in range(N):
        # Get neighbors of spot i
        neighbor_indices = indices_list[i]

        # Skip self and keep only j > i (upper triangle)
        for j in neighbor_indices:
            if j > i:  # Only upper triangle to avoid duplicates
                # Compute similarity (cosine)
                if issparse(X_norm):
                    if hasattr(X_norm[i], 'toarray'):
                        sim = X_norm[i].dot(X_norm[j].T)
                        if hasattr(sim, 'toarray'):
                            sim = sim.toarray()[0, 0]
                        else:
                            sim = sim
                    else:
                        sim = X_norm[i].dot(X_norm[j].T)
                else:
                    sim = np.dot(X_norm[i], X_norm[j])

                all_i.append(i)
                all_j.append(j)
                all_weights.append(sim)

    if len(all_i) == 0:
        if verbose:
            print("No spatial pairs found within radius")
        return None

    # Convert to arrays
    i_idx = np.array(all_i, dtype=np.int64)
    j_idx = np.array(all_j, dtype=np.int64)
    weights = np.array(all_weights, dtype=np.float64)

    # Filter by similarity threshold
    valid_mask = weights >= th_spatial
    i_idx = i_idx[valid_mask]
    j_idx = j_idx[valid_mask]
    weights = weights[valid_mask]

    if verbose:
        print(f"Found {len(i_idx)} spatial pairs (after filtering by th_spatial={th_spatial})")

    if len(i_idx) == 0:
        if verbose:
            print("No pairs passed similarity threshold")
        return None

    return {
        'i': i_idx,
        'j': j_idx,
        'w': weights
    }