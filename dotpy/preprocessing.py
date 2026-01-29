"""
Preprocessing utilities for DOT algorithm

Functions for processing reference single-cell and spatial transcriptomics data.
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict, List
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix
from sklearn.cluster import KMeans
import warnings


def setup_reference(
    adata: AnnData,
    cell_type_key: str,
    subcluster_size: int = 10,
    max_genes: int = 5000,
    remove_mt: bool = True,
    verbose: bool = False,
    device: Optional[str] = None
) -> Dict:
    """
    Process reference single-cell RNA-seq data for DOT.
    
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
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Gene expression centroids (subclusters × genes)
        - 'clusters': Dictionary mapping cell types to subcluster indices
        - 'ratios': Cell type abundance ratios
        - 'genes': Gene names
        - 'device': Device being used
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("Preprocessing reference data...")
        print(f"Using device: {device}")
    
    # Extract counts and annotations
    if issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = np.array(adata.X)
    
    annotations = adata.obs[cell_type_key].values.astype(str)
    genes = adata.var_names.values
    
    # Remove MT genes if requested
    if remove_mt:
        mt_genes = [g.startswith(('MT-', 'mt-', 'HLA-', 'RPL')) for g in genes]
        counts = counts[:, ~np.array(mt_genes)]
        genes = genes[~np.array(mt_genes)]
        if verbose:
            print(f"Removed {sum(mt_genes)} mitochondrial genes")
    
    # Select variable genes
    if counts.shape[1] > max_genes:
        if verbose:
            print(f"Selecting {max_genes} variable genes...")
        gene_var = np.var(counts, axis=0)
        top_genes = np.argsort(gene_var)[-max_genes:]
        counts = counts[:, top_genes]
        genes = genes[top_genes]
    
    # Remove empty cells
    cell_sums = counts.sum(axis=1)
    nonempty = cell_sums > 0
    counts = counts[nonempty]
    annotations = annotations[nonempty]
    
    if verbose:
        print("Aggregating cell types...")
    
    # Aggregate and subcluster
    ref_agg = _aggregate_reference(
        counts, annotations, subcluster_size, verbose=verbose
    )
    
    # Select DE genes
    if verbose:
        print("Selecting differentially expressed genes...")
    de_genes = _get_de_genes(
        ref_agg['sub_centroids'], 
        ref_agg['sub_ratios'],
        max_genes,
        verbose=verbose
    )
    
    # Prepare output
    result = {
        'X': torch.tensor(
            ref_agg['sub_centroids'][:, de_genes], 
            dtype=torch.float32,
            device=device
        ),
        'clusters': ref_agg['clusters'],
        'ratios': ref_agg['major_ratios'],
        'genes': genes[de_genes],
        'device': device
    }
    
    if verbose:
        print(f"Reference prepared: {result['X'].shape[0]} subclusters, "
              f"{result['X'].shape[1]} genes")
    
    return result


def setup_spatial(
    adata: AnnData,
    spatial_key: str = 'spatial',
    th_spatial: float = 0.84,
    th_nonspatial: float = 0.0,
    th_gene_low: float = 0.01,
    th_gene_high: float = 0.99,
    remove_mt: bool = True,
    radius: Union[str, float] = 'auto',
    verbose: bool = False,
    device: Optional[str] = None
) -> Dict:
    """
    Process spatial transcriptomics data for DOT.
    
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
    radius : float or 'auto'
        Spatial neighborhood radius
    verbose : bool
        Print progress messages
    device : str, optional
        Device for PyTorch tensors
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Gene expression matrix (spots × genes)
        - 'coords': Spatial coordinates
        - 'pairs': Spatial neighbor pairs (if th_spatial > 0)
        - 'genes': Gene names
        - 'device': Device being used
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("Preprocessing spatial data...")
        print(f"Using device: {device}")
    
    # Extract counts
    if issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = np.array(adata.X)
    
    genes = adata.var_names.values
    
    # Extract coordinates
    if spatial_key in adata.obsm:
        coords = np.array(adata.obsm[spatial_key])
    else:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
    
    # Take first 2 columns if more than 2
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    
    # Remove MT genes
    if remove_mt:
        mt_genes = [g.startswith(('MT-', 'mt-', 'HLA-', 'RPL')) for g in genes]
        counts = counts[:, ~np.array(mt_genes)]
        genes = genes[~np.array(mt_genes)]
    
    # Filter genes by expression frequency
    gene_freq = (counts > 0).mean(axis=0)
    valid_genes = (gene_freq > th_gene_low) & (gene_freq < th_gene_high)
    counts = counts[:, valid_genes]
    genes = genes[valid_genes]
    
    if verbose:
        print(f"Filtered to {counts.shape[1]} genes")
    
    # Convert to tensors
    result = {
        'X': torch.tensor(counts, dtype=torch.float32, device=device),
        'coords': torch.tensor(coords, dtype=torch.float32, device=device),
        'genes': genes,
        'device': device
    }
    
    # Find spatial pairs if requested
    if th_spatial > 0:
        if verbose:
            print("Finding spatial neighbor pairs...")
        pairs = _get_spatial_pairs(
            result['X'], 
            result['coords'],
            th_spatial,
            th_nonspatial,
            radius,
            verbose
        )
        result['pairs'] = pairs
    
    if verbose:
        print(f"Spatial data prepared: {result['X'].shape[0]} spots, "
              f"{result['X'].shape[1]} genes")
    
    return result


def _aggregate_reference(
    counts: np.ndarray,
    annotations: np.ndarray,
    cluster_size: int,
    th_inner_logfold: float = 0.75,
    verbose: bool = False
) -> Dict:
    """Aggregate reference data with subclustering."""
    major_types = np.unique(annotations)
    
    # Compute major type centroids and ratios
    major_centroids = []
    major_labels = []
    for ct in major_types:
        ct_mask = annotations == ct
        ct_counts = counts[ct_mask]
        # Sample if too many cells
        if ct_counts.shape[0] > 1000:
            idx = np.random.choice(ct_counts.shape[0], 1000, replace=False)
            ct_counts = ct_counts[idx]
        major_centroids.append(ct_counts.mean(axis=0))
        major_labels.append(ct)
    
    major_centroids = np.array(major_centroids)
    major_ratios = {ct: (annotations == ct).sum() for ct in major_types}
    total = sum(major_ratios.values())
    major_ratios = {k: v/total for k, v in major_ratios.items()}
    
    # Subcluster if requested
    if cluster_size > 1:
        if verbose:
            print("Subclustering cell types...")
        
        sub_centroids = []
        sub_labels = []
        clusters = {}
        
        for i, ct in enumerate(major_types):
            ct_mask = annotations == ct
            ct_counts = counts[ct_mask]
            
            if ct_counts.shape[0] <= 1:
                continue
            
            # Limit cells for clustering
            if ct_counts.shape[0] > 10000:
                idx = np.random.choice(ct_counts.shape[0], 10000, replace=False)
                ct_counts = ct_counts[idx]
            
            # Determine number of clusters
            K = min(cluster_size, max(1, int(2 * np.log(ct_counts.shape[0]) - 7)))
            
            if K <= 1:
                sub_centroids.append(ct_counts.mean(axis=0))
                sub_labels.append(f"{ct}__SC1")
                clusters[ct] = [len(sub_centroids) - 1]
            else:
                # Select genes for clustering based on DE
                if counts.shape[1] > 500:
                    this_ct = major_centroids[i]
                    other_ct = np.average(
                        major_centroids[np.arange(len(major_types)) != i],
                        axis=0,
                        weights=[major_ratios[ct2] for ct2 in major_types if ct2 != ct]
                    )
                    logfc = np.log((this_ct + 1e-9) / (other_ct + 1e-9))
                    top_genes = np.argsort(logfc)[-500:]
                    ct_counts_subset = ct_counts[:, top_genes]
                else:
                    ct_counts_subset = ct_counts
                
                # K-means clustering
                if verbose:
                    print(f"Clustering {ct_counts.shape[0]} {ct} cells into {K} clusters...")
                
                kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
                labels = kmeans.fit_predict(ct_counts_subset)
                
                # Remove rare clusters
                label_counts = np.bincount(labels)
                rare_labels = np.where(label_counts < 0.025 * len(labels))[0]
                valid_mask = ~np.isin(labels, rare_labels)
                
                ct_counts = ct_counts[valid_mask]
                labels = labels[valid_mask]
                
                # Compute centroids
                cluster_ids = []
                for k in range(K):
                    if (labels == k).sum() > 0:
                        sub_centroids.append(ct_counts[labels == k].mean(axis=0))
                        sub_labels.append(f"{ct}__SC{k+1}")
                        cluster_ids.append(len(sub_centroids) - 1)
                
                clusters[ct] = cluster_ids
        
        sub_centroids = np.array(sub_centroids)
        sub_ratios = {label: 1.0/len(sub_labels) for label in sub_labels}
    else:
        sub_centroids = major_centroids
        sub_labels = major_types
        clusters = {ct: [i] for i, ct in enumerate(major_types)}
        sub_ratios = major_ratios.copy()
    
    return {
        'major_centroids': major_centroids,
        'major_ratios': major_ratios,
        'sub_centroids': sub_centroids,
        'sub_ratios': sub_ratios,
        'clusters': clusters
    }


def _get_de_genes(
    centroids: np.ndarray,
    ratios: Dict,
    max_genes: int,
    verbose: bool = False
) -> np.ndarray:
    """Select differentially expressed genes."""
    if centroids.shape[1] <= max_genes:
        return np.arange(centroids.shape[1])
    
    C, G = centroids.shape
    centroids = centroids + 1e-9
    
    # Compute gene scores based on rank
    gene_scores = np.zeros((C, G))
    
    for i in range(C):
        this_ct = np.tile(centroids[i], (C-1, 1))
        other_ct = centroids[np.arange(C) != i]
        logfc = np.log(this_ct / other_ct)
        
        # Rank genes for each comparison
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, logfc)
        ranks = ranks.max() - ranks + 1
        gene_scores[i] = np.median(ranks, axis=0)
    
    # Select genes with best minimum scores across cell types
    min_scores = gene_scores.min(axis=0)
    top_genes = np.argsort(min_scores)[:max_genes]
    
    if verbose:
        print(f"Selected {len(top_genes)} DE genes")
    
    return top_genes


def _get_spatial_pairs(
    X: torch.Tensor,
    coords: torch.Tensor,
    th_spatial: float,
    th_nonspatial: float,
    radius: Union[str, float],
    verbose: bool = False
) -> Optional[Dict]:
    """Find spatial neighbor pairs based on similarity and distance."""
    # Normalize for similarity computation
    X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-10)
    
    # Compute pairwise distances
    if radius == 'auto':
        # Estimate radius from nearest neighbors
        dists = torch.cdist(coords, coords)
        k = min(8, coords.shape[0] - 1)
        knn_dists, _ = torch.topk(dists, k+1, largest=False, dim=1)
        radius = torch.quantile(knn_dists[:, 1:].flatten(), 0.9).item() * 1.05
        if verbose:
            print(f"Estimated spatial radius: {radius:.2f}")
    
    # Find spatial neighbors
    dists = torch.cdist(coords, coords)
    spatial_mask = (dists < radius) & (dists > 0)
    
    # Compute similarity
    similarity = X_norm @ X_norm.T
    
    # Filter by spatial proximity and similarity
    valid_mask = spatial_mask & (similarity >= th_spatial)
    
    # Get pairs
    i_idx, j_idx = torch.where(valid_mask & (torch.triu(torch.ones_like(valid_mask), diagonal=1) == 1))
    
    if len(i_idx) == 0:
        if verbose:
            print("No spatial pairs found")
        return None
    
    weights = similarity[i_idx, j_idx]
    
    # Add non-spatial pairs if requested
    if th_nonspatial > 0:
        nonspatial_mask = ~spatial_mask & (similarity >= th_nonspatial)
        i_ns, j_ns = torch.where(nonspatial_mask & (torch.triu(torch.ones_like(nonspatial_mask), diagonal=1) == 1))
        
        if len(i_ns) > 0:
            # Combine
            i_idx = torch.cat([i_idx, i_ns])
            j_idx = torch.cat([j_idx, j_ns])
            weights = torch.cat([weights, similarity[i_ns, j_ns]])
    
    if verbose:
        print(f"Found {len(i_idx)} spatial pairs")
    
    return {
        'i': i_idx,
        'j': j_idx,
        'w': weights
    }
