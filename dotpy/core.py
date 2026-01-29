"""
Improved core DOT algorithm with batched optimization and checkpointing

Memory-efficient implementation that processes data in batches and supports
resuming from checkpoints.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from scipy.sparse import issparse, csr_matrix
import time
import pickle
from pathlib import Path


class DOT:
    """
    Deconvolution by Optimal Transport with batched optimization.
    
    Memory-efficient implementation that:
    - Processes data in batches to reduce GPU memory usage
    - Supports checkpointing for resuming from failures
    - Only loads data to GPU when needed
    - Keeps reference data on CPU and streams to GPU
    
    Attributes
    ----------
    ref : dict
        Processed reference data from setup_reference()
    spatial : dict
        Processed spatial data from setup_spatial()
    weights : torch.Tensor
        Cell type weights per spot (spots × cell_types)
    solution : torch.Tensor
        Raw solution matrix (subclusters × spots)
    history : dict
        Optimization history
    """
    
    def __init__(
        self,
        spatial: Dict,
        ref: Dict,
        ls_solution: bool = True,
        batch_size: int = 500
    ):
        """
        Initialize DOT object.
        
        Parameters
        ----------
        spatial : dict
            Processed spatial data from setup_spatial()
        ref : dict
            Processed reference data from setup_reference()
        ls_solution : bool
            Whether to compute initial least squares solution
        batch_size : int
            Batch size for processing spots (reduces GPU memory)
        """
        # Align genes
        common_genes = np.intersect1d(spatial['genes'], ref['genes'])
        
        if len(common_genes) == 0:
            raise ValueError("No common genes found between spatial and reference data")
        
        # Get gene indices
        spatial_gene_idx = np.where(np.isin(spatial['genes'], common_genes))[0]
        ref_gene_idx = np.where(np.isin(ref['genes'], common_genes))[0]
        
        # Store data (keep sparse if possible)
        X_spatial = spatial['X_sparse']
        X_ref = ref['X_sparse']
        
        # Subset to common genes
        if issparse(X_spatial):
            X_spatial = X_spatial[:, spatial_gene_idx]
        else:
            X_spatial = X_spatial[:, spatial_gene_idx]
        
        if issparse(X_ref):
            X_ref = X_ref[:, ref_gene_idx]
        else:
            X_ref = X_ref[:, ref_gene_idx]
        
        self.spatial = {
            'X_sparse': X_spatial,
            'coords': spatial['coords'],
            'genes': spatial['genes'][spatial_gene_idx],
            'device': spatial['device']
        }
        
        if 'pairs' in spatial:
            self.spatial['pairs'] = spatial['pairs']
        
        self.ref = {
            'X_sparse': X_ref,
            'clusters': ref['clusters'],
            'ratios': ref['ratios'],
            'genes': ref['genes'][ref_gene_idx],
            'device': ref['device']
        }
        
        self.batch_size = batch_size
        self.solution = None
        self.weights = None
        self.history = None
        
        if ls_solution:
            self.solution = self._ls_solution()
    
    def fit(
        self,
        mode: str = 'highres',
        ratios_weight: float = 0.0,
        max_spot_size: int = 20,
        iterations: int = 100,
        gap_threshold: float = 0.01,
        verbose: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 10,
        resume_from: Optional[str] = None
    ) -> 'DOT':
        """
        Run the DOT algorithm with batched optimization.
        
        Parameters
        ----------
        mode : str
            'highres' for high-resolution or 'lowres' for low-resolution data
        ratios_weight : float
            Weight for matching cell type abundance (0-1)
        max_spot_size : int
            Maximum number of cells per spot
        iterations : int
            Maximum number of Frank-Wolfe iterations
        gap_threshold : float
            Convergence threshold on relative duality gap
        verbose : bool
            Print optimization progress
        checkpoint_dir : str, optional
            Directory to save checkpoints
        checkpoint_freq : int
            Save checkpoint every N iterations
        resume_from : str, optional
            Path to checkpoint to resume from
            
        Returns
        -------
        self : DOT
            Fitted DOT object
        """
        if mode == 'highres':
            sparsity_coef = 0.6
            max_size = 1
        elif mode == 'lowres':
            sparsity_coef = 0.4
            max_size = max_spot_size
        else:
            raise ValueError("mode must be 'highres' or 'lowres'")
        
        # Resume from checkpoint if provided
        start_iteration = 1
        if resume_from is not None:
            start_iteration = self._load_checkpoint(resume_from, verbose)
        
        self._run_optimization_batched(
            ratios_weight=ratios_weight,
            sparsity_coef=sparsity_coef,
            max_size=max_size,
            min_size=1,
            iterations=iterations,
            gap_threshold=gap_threshold,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            start_iteration=start_iteration
        )
        
        return self
    
    def _ls_solution(self, lambda_ridge: float = 100.0) -> np.ndarray:
        """Compute initial least squares solution."""
        # Convert to dense for LS (still on CPU)
        if issparse(self.ref['X_sparse']):
            X_ref = self.ref['X_sparse'].toarray()
        else:
            X_ref = self.ref['X_sparse']
        
        if issparse(self.spatial['X_sparse']):
            X_spatial = self.spatial['X_sparse'].toarray()
        else:
            X_spatial = self.spatial['X_sparse']
        
        C = X_ref.shape[0]
        
        # (X^T X + λI)^{-1} X^T Y
        XtX = X_ref @ X_ref.T
        XtX += np.eye(C) * lambda_ridge
        
        XtY = X_ref @ X_spatial.T
        
        solution = np.linalg.solve(XtX, XtY)
        solution = np.maximum(solution, 0)
        
        return solution
    
    def _run_optimization_batched(
        self,
        ratios_weight: float,
        sparsity_coef: float,
        max_size: int,
        min_size: int,
        iterations: int,
        gap_threshold: float,
        verbose: bool,
        checkpoint_dir: Optional[str],
        checkpoint_freq: int,
        start_iteration: int = 1
    ):
        """
        Main Frank-Wolfe optimization with batching.
        
        Processes spots in batches to reduce GPU memory usage.
        """
        device = self.ref['device']
        
        # Convert sparse matrices to dense on CPU
        if issparse(self.ref['X_sparse']):
            X_ref_cpu = torch.from_numpy(self.ref['X_sparse'].toarray()).float()
        else:
            X_ref_cpu = torch.from_numpy(self.ref['X_sparse']).float()
        
        if issparse(self.spatial['X_sparse']):
            X_spatial_cpu = torch.from_numpy(self.spatial['X_sparse'].toarray()).float()
        else:
            X_spatial_cpu = torch.from_numpy(self.spatial['X_sparse']).float()
        
        # Data dimensions
        S = X_spatial_cpu.shape[0]  # spots
        C = X_ref_cpu.shape[0]  # subclusters
        G = X_spatial_cpu.shape[1]  # genes
        K = len(self.ref['clusters'])  # major cell types
        
        # Create cluster mapping
        cluster_to_major = torch.zeros(C, dtype=torch.long)
        for k, (ct, indices) in enumerate(self.ref['clusters'].items()):
            cluster_to_major[indices] = k
        
        # Cell type ratios
        cell_types = list(self.ref['clusters'].keys())
        sc_ratios = torch.tensor(
            [self.ref['ratios'][ct] for ct in cell_types],
            dtype=torch.float32
        )
        sc_ratios = sc_ratios / sc_ratios.sum()
        
        # Expected counts per spot
        r_st = torch.full((S,), 0.9 * min_size + 0.1 * max_size)
        n_st = r_st.sum()
        r_sc = sc_ratios * n_st
        r_sc_ex = r_sc[cluster_to_major]
        
        # Loss weights
        inner_params = [1.0, 0.25 if max_size > 1 else 1.0, 0.0, 0.01]
        
        l_a = ratios_weight / max_size
        l_g = inner_params[0] * S / G
        l_i = inner_params[1]
        l_sp = l_i * sparsity_coef / max_size
        
        # Spatial pairs
        has_pairs = 'pairs' in self.spatial and self.spatial['pairs'] is not None
        if has_pairs:
            l_s = inner_params[3] * S / (max_size * len(self.spatial['pairs']['i']))
            pairs_i = torch.from_numpy(self.spatial['pairs']['i']).long()
            pairs_j = torch.from_numpy(self.spatial['pairs']['j']).long()
            pairs_w = torch.from_numpy(self.spatial['pairs']['w']).float()
        else:
            l_s = 0.0
        
        # Precompute normalized reference on CPU
        X_ref_norm_cpu = F.normalize(X_ref_cpu, p=2, dim=1)
        
        # Initialize solution on CPU
        if self.solution is None:
            # Initialize with cell type ratios
            initial_ratios = torch.zeros(C)
            for k, ct in enumerate(cell_types):
                indices = self.ref['clusters'][ct]
                initial_ratios[indices] = sc_ratios[k] / len(indices)
            
            Yt = initial_ratios.unsqueeze(1) * r_st.unsqueeze(0)
        else:
            Yt = torch.from_numpy(self.solution).float()
            Yt = torch.clamp(Yt, min=0)
        
        # Optimization loop
        f_best = float('inf')
        lb = float('-inf')
        Y_best = None
        
        history = {
            'iteration': [],
            'objective': [],
            'upper_bound': [],
            'lower_bound': [],
            'gap': [],
            'time': []
        }
        
        lg2 = np.log(2)
        n_batches = int(np.ceil(S / self.batch_size))
        
        if verbose:
            print(f"Starting optimization with {n_batches} batches of {self.batch_size} spots")
        
        for iteration in range(start_iteration, iterations + 1):
            iter_start = time.time()
            
            # Aggregate to major cell types
            Ytk = torch.zeros(K, S)
            for k, ct in enumerate(cell_types):
                indices = self.ref['clusters'][ct]
                Ytk[k] = Yt[indices].sum(dim=0)
            
            rho_tk = Ytk.sum(dim=1)
            rho_tk = torch.clamp(rho_tk, min=1e-10)
            rho_t_ex = rho_tk[cluster_to_major]
            
            # Initialize gradient on CPU
            Dt = torch.zeros_like(Yt)
            
            # Abundance matching term
            ratio_error = 0.0
            if l_a > 0:
                rho_avg = (rho_tk + r_sc) / 2
                log_rho = 0.5 * self._safe_log2(rho_tk / rho_avg)
                log_rsc = 0.5 * self._safe_log2(r_sc / rho_avg)
                
                ratio_error = (rho_tk * log_rho).sum() + (r_sc * log_rsc).sum()
                
                log_rho_ex = log_rho[cluster_to_major]
                d_ratio = l_a * log_rho_ex
                
                Dt += d_ratio.unsqueeze(1).expand_as(Yt)
            
            # Gene expression matching - BATCHED
            dcosine_st = 0.0
            dcosine_g = 0.0
            dcosine_lin = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, S)
                
                # Move batch to GPU
                Yt_batch = Yt[:, start_idx:end_idx].to(device)
                X_spatial_batch = X_spatial_cpu[start_idx:end_idx].to(device)
                X_ref_gpu = X_ref_cpu.to(device)
                X_ref_norm_gpu = X_ref_norm_cpu.to(device)
                
                # Predicted expression for batch
                st_xt = Yt_batch.T @ X_ref_gpu
                
                # Spot-wise cosine distance
                if sparsity_coef < 1 and l_i > 0:
                    st_xt_norms = torch.norm(st_xt, dim=1, keepdim=True)
                    st_xt_n = st_xt / (st_xt_norms + 1e-10)
                    
                    csi = (X_spatial_batch * st_xt_n).sum(dim=1)
                    di = 1 - csi
                    di = torch.clamp(di, min=0)
                    
                    d_i_grad = 1.0 / (2 * torch.sqrt(di) + 1e-10)
                    di = torch.sqrt(di)
                    
                    dcosine_st += di.sum().item()
                    
                    st_de = l_i * (1 - sparsity_coef) * (
                        X_spatial_batch - st_xt_n * csi.unsqueeze(1)
                    ) * d_i_grad.unsqueeze(1) / (st_xt_norms + 1e-10)
                    
                    Dt[:, start_idx:end_idx] -= (st_de @ X_ref_gpu.T).T.cpu()
                
                # Gene-wise cosine (aggregate across batches)
                if l_g > 0:
                    # Accumulate for gene-wise computation
                    pass  # Will compute after batches
                
                # Linear sparsity
                if l_sp > 0:
                    X_spatial_norm = F.normalize(X_spatial_batch, p=2, dim=1)
                    linear_dcosine_batch = 1 - (X_ref_norm_gpu @ X_spatial_norm.T)
                    linear_dcosine_batch = torch.clamp(linear_dcosine_batch, min=0)
                    linear_dcosine_batch = torch.sqrt(linear_dcosine_batch)
                    
                    Dt[:, start_idx:end_idx] += (l_sp * linear_dcosine_batch).cpu()
                    dcosine_lin += (Yt_batch * linear_dcosine_batch).sum().item()
                
                # Clear GPU memory
                del Yt_batch, X_spatial_batch, X_ref_gpu, X_ref_norm_gpu
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Gene-wise cosine (if needed, compute on full data)
            if l_g > 0:
                # Move full data to GPU temporarily
                X_ref_gpu = X_ref_cpu.to(device)
                st_xt_full = Yt.T.to(device) @ X_ref_gpu
                X_spatial_gpu = X_spatial_cpu.to(device)
                
                st_xg = F.normalize(X_spatial_gpu, p=2, dim=0)
                st_xt_gnorms = torch.norm(st_xt_full, dim=0, keepdim=True)
                st_xt_gn = st_xt_full / (st_xt_gnorms + 1e-10)
                
                csg = (st_xt_gn * st_xg).sum(dim=0)
                dg = 1 - csg
                dg = torch.clamp(dg, min=0)
                
                dg_coefs = 1.0 / (2 * torch.sqrt(dg) + 1e-10) / (st_xt_gnorms + 1e-10)
                dg = torch.sqrt(dg)
                
                dcosine_g = dg.sum().item()
                
                st_de_g = l_g * (st_xg - st_xt_gn * csg.unsqueeze(0)) * dg_coefs
                Dt -= (st_de_g @ X_ref_gpu.T).T.cpu()
                
                # Clear GPU
                del X_ref_gpu, st_xt_full, X_spatial_gpu
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Spatial coherence (on CPU to save GPU memory)
            d_s = 0.0
            if l_s > 0 and has_pairs:
                Dtk = torch.zeros_like(Ytk)
                
                for p in range(len(pairs_i)):
                    i = pairs_i[p]
                    j = pairs_j[p]
                    w = pairs_w[p] * 0.5 / lg2
                    
                    ym = 0.5 * (Ytk[:, i] + Ytk[:, j])
                    
                    for ii in [i, j]:
                        l_ii = self._safe_log2(Ytk[:, ii] / (ym + 1e-10))
                        d_s += w * (Ytk[:, ii] * l_ii).sum().item()
                        Dtk[:, ii] += l_s * w * l_ii
                
                # Map back to subclusters
                for k, ct in enumerate(cell_types):
                    indices = self.ref['clusters'][ct]
                    Dt[indices] += Dtk[k].unsqueeze(0)
            
            # Frank-Wolfe step
            Yt_h = torch.zeros_like(Yt)
            for i in range(S):
                kk = torch.argmin(Dt[:, i])
                Yt_h[kk, i] = max_size if Dt[kk, i] < 0 else min_size
            
            # Compute objective and gap
            ft = (l_i * (1 - sparsity_coef) * dcosine_st + 
                  l_sp * dcosine_lin + 
                  l_g * dcosine_g + 
                  l_s * d_s + 
                  l_a * ratio_error)
            
            gap = (Dt * (Yt - Yt_h)).sum().item()
            
            if ft < f_best:
                f_best = ft
                Y_best = Yt.clone()
            
            lb = max(lb, ft - gap)
            
            rel_gap = gap / abs(f_best) if abs(f_best) > 1e-10 else gap
            
            # Step size
            step = min(0.99, 2.0 / (iteration + 1))
            
            iter_time = time.time() - iter_start
            
            if verbose and iteration % 10 == 1:
                print(f"Iter {iteration:3d}: obj={ft:8.4f}, gap={rel_gap:6.4f}, "
                      f"time={iter_time:5.2f}s")
            
            # Store history
            history['iteration'].append(iteration)
            history['objective'].append(ft)
            history['upper_bound'].append(f_best)
            history['lower_bound'].append(lb if lb != float('-inf') else None)
            history['gap'].append(rel_gap)
            history['time'].append(iter_time)
            
            # Save checkpoint
            if checkpoint_dir is not None and iteration % checkpoint_freq == 0:
                self._save_checkpoint(
                    checkpoint_dir,
                    iteration,
                    Yt,
                    Y_best,
                    f_best,
                    lb,
                    history,
                    verbose
                )
            
            # Check convergence
            if rel_gap <= gap_threshold and iteration >= 10:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            if step <= 1e-5:
                if verbose:
                    print(f"Step size too small at iteration {iteration}")
                break
            
            # Update solution
            Yt = Yt - step * (Yt - Yt_h)
        
        # Store results
        self.solution = Y_best.numpy()
        
        # Compute cell type weights
        weights = np.zeros((S, K))
        for k, ct in enumerate(cell_types):
            indices = self.ref['clusters'][ct]
            weights[:, k] = Y_best[indices].sum(dim=0).numpy()
        
        self.weights = weights
        self.history = history
        
        if verbose:
            print(f"\nOptimization complete. Final objective: {f_best:.4f}")
    
    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        iteration: int,
        Yt: torch.Tensor,
        Y_best: torch.Tensor,
        f_best: float,
        lb: float,
        history: dict,
        verbose: bool
    ):
        """Save optimization checkpoint."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_iter_{iteration}.pkl"
        
        checkpoint = {
            'iteration': iteration,
            'Yt': Yt.numpy(),
            'Y_best': Y_best.numpy() if Y_best is not None else None,
            'f_best': f_best,
            'lb': lb,
            'history': history,
            'solution': self.solution
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if verbose:
            print(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str, verbose: bool) -> int:
        """Load optimization checkpoint and resume."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.solution = checkpoint['Y_best']
        self.history = checkpoint['history']
        
        if verbose:
            print(f"Resumed from checkpoint: {checkpoint_path}")
            print(f"Starting from iteration {checkpoint['iteration'] + 1}")
        
        return checkpoint['iteration'] + 1
    
    @staticmethod
    def _safe_log2(x: torch.Tensor) -> torch.Tensor:
        """Safe log2 that handles zeros."""
        log_x = torch.log2(torch.clamp(x, min=1e-10))
        log_x[torch.isnan(log_x)] = 0
        log_x[torch.isinf(log_x)] = -20
        return log_x
    
    def get_weights(self, normalize: bool = True) -> np.ndarray:
        """
        Get cell type weights as numpy array.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize weights to sum to 1 per spot
            
        Returns
        -------
        weights : np.ndarray
            Cell type weights (spots × cell_types)
        """
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        weights = self.weights.copy()
        
        if normalize:
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            weights = weights / row_sums
        
        return weights
    
    def get_cell_types(self) -> list:
        """Get list of cell type names."""
        return list(self.ref['clusters'].keys())
