"""
Preprocessing utilities for DOT algorithm - R-compatible version

Memory-efficient implementation with sparse matrix support.
"""

import numpy as np
from typing import Optional, Dict
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix, vstack
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors
import scanpy as sc


def _select_kmeans_genes(
    ct_centroid: np.ndarray,
    major_centroids: np.ndarray,
    major_ratios: Dict[str, float],
    ct_name: str,
    th_logfold: float = 0.75,
    max_genes: int = 500
) -> np.ndarray:
    """Select genes for k-means clustering based on log-fold change."""
    all_types = list(major_ratios.keys())
    ct_idx = all_types.index(ct_name)
    other_types = [t for i, t in enumerate(all_types) if i != ct_idx]

    if len(other_types) == 0:
        return np.arange(len(ct_centroid))

    other_ratios_norm = np.array([major_ratios[t] for t in other_types])
    other_ratios_norm = other_ratios_norm / other_ratios_norm.sum()

    other_indices = [i for i, t in enumerate(all_types) if t in other_types]
    other_centroids = major_centroids[other_indices, :]

    other_avg = other_ratios_norm @ other_centroids
    logfc = np.log((ct_centroid + 1e-9) / (other_avg + 1e-9))

    passing = np.where(logfc > th_logfold)[0]

    if len(passing) == 0:
        return np.arange(len(ct_centroid))

    if len(passing) > max_genes:
        passing = passing[np.argsort(logfc[passing])[::-1][:max_genes]]

    return passing


def _kmeans_subcluster(
    X_ct: np.ndarray,
    gene_indices: np.ndarray,
    K: int,
    min_frac: float = 0.025,
    random_state: int = 42
) -> np.ndarray:
    """K-means clustering with small cluster filtering."""
    if issparse(X_ct):
        X_subset = X_ct[:, gene_indices].toarray()
    else:
        X_subset = X_ct[:, gene_indices]

    if X_subset.shape[0] > 10000:
        km = MiniBatchKMeans(
            n_clusters=K,
            n_init=10,
            random_state=random_state,
            batch_size=1000,
            max_iter=300
        )
    else:
        km = KMeans(
            n_clusters=K,
            n_init=10,
            random_state=random_state,
            max_iter=300
        )

    labels = km.fit_predict(X_subset)

    n_cells = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    noise_mask = counts < (min_frac * n_cells)
    noise_clusters = unique_labels[noise_mask]

    if len(noise_clusters) > 0:
        labels = labels.copy()
        for nc in noise_clusters:
            labels[labels == nc] = -1

    return labels


def _get_de_genes_r_style(
    centroids: np.ndarray,
    max_genes: int,
    verbose: bool = False
) -> np.ndarray:
    """Select DE genes using R's median rank scoring method."""
    if centroids.shape[1] <= max_genes:
        return np.arange(centroids.shape[1])

    C, G = centroids.shape

    if issparse(centroids):
        centroids_dense = centroids.toarray() + 1e-9
    else:
        centroids_dense = centroids.copy() + 1e-9

    gene_scores = np.zeros((C, G), dtype=np.float32)

    if verbose:
        print(f"Computing gene scores for {C} clusters, {G} genes...")

    for i in range(C):
        this_ct = np.tile(centroids_dense[i:i+1, :], (C-1, 1))
        other_ct = np.delete(centroids_dense, i, axis=0)
        logfc = np.log(this_ct / other_ct)

        ranks = np.empty_like(logfc)
        for j in range(C-1):
            ranks[j, :] = G - np.argsort(np.argsort(logfc[j, :]))

        gene_scores[i, :] = np.median(ranks, axis=0)

    min_scores = gene_scores.min(axis=0)
    top_genes = np.argsort(min_scores)[:max_genes]

    if verbose:
        print(f"Selected {len(top_genes)} DE genes using R-style ranking")

    return top_genes


def _aggregate_reference(
    X: np.ndarray,
    annotations: np.ndarray,
    cluster_size: int,
    th_inner_logfold: float = 0.75,
    random_state: int = 42,
    verbose: bool = False
) -> Dict:
    """Aggregate reference data with R-compatible algorithm."""
    np.random.seed(random_state)

    major_types = np.unique(annotations)
    n_genes = X.shape[1]

    if verbose:
        print("Computing major centroids...")

    candidate_indices = []
    candidate_types = []

    for ct in major_types:
        ct_idx = np.where(annotations == ct)[0]

        if len(ct_idx) > 1000:
            ct_idx = np.random.choice(ct_idx, 1000, replace=False)

        candidate_indices.extend(ct_idx)
        candidate_types.extend([ct] * len(ct_idx))

    candidate_indices = np.array(candidate_indices)
    candidate_types = np.array(candidate_types)

    major_centroids_list = []
    major_ratios = {}

    for ct in major_types:
        ct_mask = candidate_types == ct

        if issparse(X):
            centroid = np.asarray(X[candidate_indices[ct_mask]].mean(axis=0)).flatten()
        else:
            centroid = X[candidate_indices[ct_mask]].mean(axis=0)

        major_centroids_list.append(centroid)
        major_ratios[ct] = int((annotations == ct).sum())

    total = sum(major_ratios.values())
    major_ratios = {k: v / total for k, v in major_ratios.items()}

    major_centroids = np.array(major_centroids_list)

    if cluster_size <= 1:
        if issparse(X):
            sub_centroids = vstack([csr_matrix(c) for c in major_centroids_list])
        else:
            sub_centroids = major_centroids

        clusters = {ct: [i] for i, ct in enumerate(major_types)}

        return {
            'major_centroids': major_centroids,
            'major_ratios': major_ratios,
            'sub_centroids': sub_centroids,
            'clusters': clusters
        }

    if verbose:
        print(f"Sub-clustering {len(major_types)} cell types...")

    sub_centroids_list = []
    clusters = {}

    for ct_idx, ct in enumerate(major_types):
        ct_mask = annotations == ct
        ct_indices = np.where(ct_mask)[0]

        if len(ct_indices) <= 1:
            continue

        if len(ct_indices) > 10000:
            ct_indices = np.random.choice(ct_indices, 10000, replace=False)

        X_ct = X[ct_indices]

        K = min(cluster_size, max(1, int(np.round(2 * np.log(len(ct_indices)) - 7))))

        if K <= 1:
            if issparse(X_ct):
                centroid = np.asarray(X_ct.mean(axis=0)).flatten()
            else:
                centroid = X_ct.mean(axis=0)
            sub_centroids_list.append(centroid)
            clusters[ct] = [len(sub_centroids_list) - 1]
            continue

        kmeans_genes = np.arange(n_genes)

        if n_genes > 500:
            kmeans_genes = _select_kmeans_genes(
                ct_centroid=major_centroids[ct_idx],
                major_centroids=major_centroids,
                major_ratios=major_ratios,
                ct_name=ct,
                th_logfold=th_inner_logfold,
                max_genes=500
            )

        if verbose:
            print(f"  Clustering {len(ct_indices)} {ct} cells into ~{K} clusters "
                  f"(using {len(kmeans_genes)} genes)...")

        labels = _kmeans_subcluster(
            X_ct=X_ct,
            gene_indices=kmeans_genes,
            K=K,
            min_frac=0.025,
            random_state=random_state
        )

        valid_mask = labels >= 0

        if valid_mask.sum() == 0:
            if issparse(X_ct):
                centroid = np.asarray(X_ct.mean(axis=0)).flatten()
            else:
                centroid = X_ct.mean(axis=0)
            sub_centroids_list.append(centroid)
            clusters[ct] = [len(sub_centroids_list) - 1]
            continue

        X_ct_valid = X_ct[valid_mask]
        labels_valid = labels[valid_mask]

        cluster_ids = []
        for sc_label in np.unique(labels_valid):
            sc_mask = labels_valid == sc_label

            if issparse(X_ct_valid):
                sc_centroid = np.asarray(X_ct_valid[sc_mask].mean(axis=0)).flatten()
            else:
                sc_centroid = X_ct_valid[sc_mask].mean(axis=0)

            sub_centroids_list.append(sc_centroid)
            cluster_ids.append(len(sub_centroids_list) - 1)

        clusters[ct] = cluster_ids

    if issparse(X):
        sub_centroids = vstack([csr_matrix(c) for c in sub_centroids_list])
    else:
        sub_centroids = np.array(sub_centroids_list)

    surviving = {ct: major_ratios[ct] for ct in clusters.keys() if ct in major_ratios}
    total = sum(surviving.values())
    if total > 0:
        major_ratios = {ct: v / total for ct, v in surviving.items()}

    if verbose:
        print(f"Created {len(sub_centroids_list)} sub-clusters from {len(clusters)} cell types")

    return {
        'major_centroids': major_centroids,
        'major_ratios': major_ratios,
        'sub_centroids': sub_centroids,
        'clusters': clusters
    }


def setup_reference(
    adata: AnnData,
    cell_type_key: str,
    subcluster_size: int = 10,
    max_genes: int = 5000,
    remove_mt: bool = True,
    th_inner_logfold: float = 0.75,
    random_state: int = 42,
    verbose: bool = False,
    copy: bool = True
) -> Dict:
    """
    Process reference single-cell RNA-seq data for DOT (R-compatible version).

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
        Whether to remove mitochondrial / ribosomal genes
    th_inner_logfold : float
        Log-fold threshold for gene selection in sub-clustering
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress messages
    copy : bool
        Whether to copy adata before processing

    Returns
    -------
    dict
        'X_sparse', 'clusters', 'ratios', 'genes'
    """
    if verbose:
        print("=" * 60)
        print("DOT Reference Preprocessing (R-compatible)")
        print("=" * 60)
        print(f"Input shape: {adata.shape}")

    if copy:
        adata = adata.copy()

    if verbose:
        print("\nRunning basic QC...")
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=1)

    if remove_mt:
        mt_mask = adata.var_names.str.startswith(('MT-', 'HLA-', 'RPL'))
        n_mt = mt_mask.sum()
        if n_mt > 0:
            adata = adata[:, ~mt_mask].copy()
            if verbose:
                print(f"Removed {n_mt} MT/HLA/RPL genes")

    X = adata.X
    annotations = adata.obs[cell_type_key].values.astype(str)
    genes = adata.var_names.values

    vg_genes = max(5000, max_genes)
    if adata.shape[1] > vg_genes:
        if verbose:
            print(f"\nSelecting {vg_genes} highly variable genes...")

        adata_hvg = adata.copy()
        adata_hvg.layers['counts'] = adata.X.copy()

        sc.pp.highly_variable_genes(
            adata_hvg,
            n_top_genes=vg_genes,
            flavor='seurat_v3',
            layer='counts',
            subset=False
        )

        hvg_genes = adata_hvg.var_names[adata_hvg.var['highly_variable']].tolist()
        adata = adata[:, hvg_genes].copy()
        X = adata.X
        genes = adata.var_names.values

    if verbose:
        print(f"\nAfter filtering: {X.shape}")
        print("\nAggregating and sub-clustering cell types...")

    ref_agg = _aggregate_reference(
        X=X,
        annotations=annotations,
        cluster_size=subcluster_size,
        th_inner_logfold=th_inner_logfold,
        random_state=random_state,
        verbose=verbose
    )

    if verbose:
        print("\nSelecting differentially expressed genes (R-style)...")

    de_genes = _get_de_genes_r_style(
        centroids=ref_agg['sub_centroids'],
        max_genes=max_genes,
        verbose=verbose
    )

    if issparse(ref_agg['sub_centroids']):
        X_subset = ref_agg['sub_centroids'][:, de_genes]
    else:
        X_subset = ref_agg['sub_centroids'][:, de_genes]

    result = {
        'X_sparse': X_subset,
        'clusters': ref_agg['clusters'],
        'ratios': ref_agg['major_ratios'],
        'genes': genes[de_genes]
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Reference prepared:")
        print(f"  - {X_subset.shape[0]} sub-clusters")
        print(f"  - {X_subset.shape[1]} genes")
        if issparse(X_subset):
            sparsity = 1 - X_subset.nnz / (X_subset.shape[0] * X_subset.shape[1])
            print(f"  - Sparsity: {sparsity:.2%}")
        print(f"{'=' * 60}")

    return result


def setup_spatial(
    adata: AnnData,
    spatial_key: str = 'spatial',
    th_spatial: float = 0.84,
    th_nonspatial: float = 0.0,
    th_gene_low: float = 0.01,
    th_gene_high: float = 0.99,
    remove_mt: bool = True,
    radius: str = 'auto',
    verbose: bool = False,
    copy: bool = True
) -> Dict:
    """
    Process spatial transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Spatial data with counts in .X
    spatial_key : str
        Key in adata.obsm containing spatial coordinates
    th_spatial : float
        Cosine similarity threshold for spatial pairs
    th_nonspatial : float
        Threshold for non-spatial pairs (0 to disable)
    th_gene_low : float
        Minimum fraction of spots a gene must be expressed in
    th_gene_high : float
        Maximum fraction of spots a gene can be expressed in
    remove_mt : bool
        Remove MT/RPL genes
    radius : str or float
        Spatial radius ('auto' or numeric value)
    verbose : bool
        Print progress
    copy : bool
        Copy adata

    Returns
    -------
    dict
        'X_sparse', 'coords', 'genes', 'pairs' (if th_spatial > 0)
    """
    if copy:
        adata = adata.copy()

    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.shape[1] > 2:
        coords = coords[:, :2]

    if verbose:
        print(f"Processing spatial data: {adata.shape}")

    # Remove MT genes
    if remove_mt:
        mt_mask = adata.var_names.str.startswith(('MT-', 'HLA-', 'RPL'))
        n_mt = mt_mask.sum()
        if n_mt > 0:
            adata = adata[:, ~mt_mask].copy()
            if verbose:
                print(f"Removed {n_mt} MT/HLA/RPL genes")

    # Gene frequency filter
    if th_gene_high < 1 or th_gene_low > 0:
        gene_freq = np.asarray((adata.X > 0).mean(axis=0)).flatten()
        valid = (gene_freq > th_gene_low) & (gene_freq < th_gene_high)
        adata = adata[:, valid].copy()
        if verbose:
            print(f"Filtered to {adata.shape[1]} genes")

    X = adata.X
    genes = adata.var_names.values

    result = {
        'X_sparse': X,
        'coords': coords,
        'genes': genes
    }

    # Spatial pairs
    if th_spatial > 0:
        if issparse(X):
            X_norm = sk_normalize(X, norm='l2', axis=1, copy=True)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_norm = X / norms

        if verbose:
            print("Finding spatial neighbours...")

        N = coords.shape[0]
        n_neighbors = 8

        if radius == 'auto':
            nbrs_est = NearestNeighbors(
                n_neighbors=min(n_neighbors + 1, N),
                algorithm='ball_tree',
                metric='euclidean'
            )
            nbrs_est.fit(coords)
            dists, _ = nbrs_est.kneighbors(coords)
            radius = float(np.quantile(dists[:, 1:].ravel(), 0.9) * 1.05)
            if verbose:
                print(f"Estimated spatial radius: {radius:.2f}")

        if verbose:
            print(f"Finding spatial neighbors (radius={radius:.2f})...")

        nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree', metric='euclidean')
        nbrs.fit(coords)
        distances_list, indices_list = nbrs.radius_neighbors(coords)

        all_i = []
        all_j = []
        for i in range(N):
            nbr = indices_list[i]
            mask = nbr > i
            js = nbr[mask]
            if len(js) > 0:
                all_i.append(np.full(len(js), i, dtype=np.int64))
                all_j.append(js.astype(np.int64))

        if len(all_i) > 0:
            i_idx = np.concatenate(all_i)
            j_idx = np.concatenate(all_j)

            # Compute cosine similarities
            PAIR_BATCH = 500_000
            n_pairs = len(i_idx)
            weights = np.empty(n_pairs, dtype=np.float64)

            for start in range(0, n_pairs, PAIR_BATCH):
                end = min(start + PAIR_BATCH, n_pairs)
                bi = i_idx[start:end]
                bj = j_idx[start:end]

                if issparse(X_norm):
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

            max_pairs = N

            if len(i_idx) > max_pairs:
                keep = np.argsort(weights)[::-1][:max_pairs]
                i_idx = i_idx[keep]
                j_idx = j_idx[keep]
                weights = weights[keep]
            elif th_nonspatial > 0 and len(i_idx) < max_pairs:
                if issparse(X_norm):
                    sim_full = np.asarray(X_norm @ X_norm.T.toarray()) if hasattr(X_norm, 'toarray') else np.asarray((X_norm @ X_norm.T).todense())
                else:
                    sim_full = X_norm @ X_norm.T

                tri_i, tri_j = np.triu_indices(N, k=1)
                tri_w = sim_full[tri_i, tri_j]
                mask = tri_w >= th_nonspatial
                cand_i = tri_i[mask]
                cand_j = tri_j[mask]
                cand_w = tri_w[mask]

                existing = set(zip(i_idx.tolist(), j_idx.tolist()))
                keep_mask = np.array([
                    (int(a), int(b)) not in existing for a, b in zip(cand_i, cand_j)
                ], dtype=bool)
                cand_i = cand_i[keep_mask]
                cand_j = cand_j[keep_mask]
                cand_w = cand_w[keep_mask]

                slots = max_pairs - len(i_idx)
                if len(cand_i) > slots:
                    top = np.argsort(cand_w)[::-1][:slots]
                    cand_i = cand_i[top]
                    cand_j = cand_j[top]
                    cand_w = cand_w[top]

                i_idx = np.concatenate([i_idx, cand_i.astype(np.int64)])
                j_idx = np.concatenate([j_idx, cand_j.astype(np.int64)])
                weights = np.concatenate([weights, cand_w.astype(np.float64)])

            if len(i_idx) > 0:
                if verbose:
                    print(f"Found {len(i_idx)} spatial pairs")

                # Return as dict
                result['pairs'] = {
                    'i': i_idx,
                    'j': j_idx,
                    'w': weights
                }
            elif verbose:
                print("No spatial pairs found")
        elif verbose:
            print("No spatial pairs found")

    if verbose:
        print(f"Spatial data prepared: {X.shape[0]} spots, {X.shape[1]} genes")
        if issparse(X):
            print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")

    return result