"""
Core DOT algorithm implementation using PyTorch

Multi-objective optimization using Frank-Wolfe algorithm for 
spatial transcriptomics deconvolution.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import time


class DOT:
    """
    Deconvolution by Optimal Transport
    
    Main class implementing the DOT algorithm for transferring cell type
    annotations from single-cell to spatial transcriptomics data.
    
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
        ls_solution: bool = True
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
        """
        # Align genes
        common_genes = np.intersect1d(spatial['genes'], ref['genes'])
        
        if len(common_genes) == 0:
            raise ValueError("No common genes found between spatial and reference data")
        
        # Get gene indices
        spatial_gene_idx = np.where(np.isin(spatial['genes'], common_genes))[0]
        ref_gene_idx = np.where(np.isin(ref['genes'], common_genes))[0]
        
        # Store data
        self.spatial = {
            'X': spatial['X'][:, spatial_gene_idx],
            'coords': spatial['coords'],
            'genes': spatial['genes'][spatial_gene_idx],
            'device': spatial['device']
        }
        
        if 'pairs' in spatial:
            self.spatial['pairs'] = spatial['pairs']
        
        self.ref = {
            'X': ref['X'][:, ref_gene_idx],
            'clusters': ref['clusters'],
            'ratios': ref['ratios'],
            'genes': ref['genes'][ref_gene_idx],
            'device': ref['device']
        }
        
        # Initialize solution
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
        verbose: bool = False
    ) -> 'DOT':
        """
        Run the DOT algorithm.
        
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
        
        self._run_optimization(
            ratios_weight=ratios_weight,
            sparsity_coef=sparsity_coef,
            max_size=max_size,
            min_size=1,
            iterations=iterations,
            gap_threshold=gap_threshold,
            verbose=verbose
        )
        
        return self
    
    def _ls_solution(self, lambda_ridge: float = 100.0) -> torch.Tensor:
        """Compute initial least squares solution."""
        C = self.ref['X'].shape[0]
        
        # (X^T X + λI)^{-1} X^T Y
        XtX = self.ref['X'] @ self.ref['X'].T
        XtX += torch.eye(C, device=self.ref['device']) * lambda_ridge
        
        XtY = self.ref['X'] @ self.spatial['X'].T
        
        solution = torch.linalg.solve(XtX, XtY)
        solution = torch.clamp(solution, min=0)
        
        return solution
    
    def _run_optimization(
        self,
        ratios_weight: float,
        sparsity_coef: float,
        max_size: int,
        min_size: int,
        iterations: int,
        gap_threshold: float,
        verbose: bool
    ):
        """Main Frank-Wolfe optimization loop."""
        device = self.ref['device']
        
        # Data dimensions
        S = self.spatial['X'].shape[0]  # spots
        C = self.ref['X'].shape[0]  # subclusters
        G = self.spatial['X'].shape[1]  # genes
        K = len(self.ref['clusters'])  # major cell types
        
        # Create cluster mapping
        cluster_to_major = torch.zeros(C, dtype=torch.long, device=device)
        for k, (ct, indices) in enumerate(self.ref['clusters'].items()):
            cluster_to_major[indices] = k
        
        # Cell type ratios
        cell_types = list(self.ref['clusters'].keys())
        sc_ratios = torch.tensor(
            [self.ref['ratios'][ct] for ct in cell_types],
            dtype=torch.float32,
            device=device
        )
        sc_ratios = sc_ratios / sc_ratios.sum()
        
        # Expected counts per spot
        r_st = torch.full((S,), 0.9 * min_size + 0.1 * max_size, device=device)
        n_st = r_st.sum()
        r_sc = sc_ratios * n_st
        r_sc_ex = r_sc[cluster_to_major]
        
        # Normalize data
        sc_xn = F.normalize(self.ref['X'], p=2, dim=1)
        st_xn = F.normalize(self.spatial['X'], p=2, dim=1)
        
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
            pairs_i = self.spatial['pairs']['i']
            pairs_j = self.spatial['pairs']['j']
            pairs_w = self.spatial['pairs']['w']
        else:
            l_s = 0.0
        
        # Precompute linear distance for sparsity
        linear_dcosine = None
        if l_sp > 0 or self.solution is None:
            linear_dcosine = 1 - (sc_xn @ st_xn.T)
            linear_dcosine = torch.clamp(linear_dcosine, min=0)
            linear_dcosine = torch.sqrt(linear_dcosine)
        
        # Initialize solution
        if self.solution is None:
            # Initialize with cell type ratios
            initial_ratios = torch.zeros(C, device=device)
            for k, ct in enumerate(cell_types):
                indices = self.ref['clusters'][ct]
                initial_ratios[indices] = sc_ratios[k] / len(indices)
            
            Yt = initial_ratios.unsqueeze(1) * r_st.unsqueeze(0)
            
            # Mix with closest match
            mix_weight = 0.1
            Yt = Yt * mix_weight
            for i in range(S):
                c_min = torch.argmin(linear_dcosine[:, i])
                Yt[c_min, i] += (1 - mix_weight) * r_st[i]
        else:
            Yt = self.solution.clone()
            # Ensure valid
            Yt = torch.clamp(Yt, min=0)
            cs = Yt.sum(dim=0)
            Yt[:, cs < 1e-3] = 1.0 / C
            
            # Adjust to size constraints
            cs_factors = torch.ones(S, device=device)
            if sparsity_coef > 0.5:
                cs_high = cs >= 1e-3
                cs_factors[cs_high] = 1.0 / cs[cs_high]
            else:
                cs_high = cs > max_size
                cs_factors[cs_high] = max_size / cs[cs_high]
                cs_low = (cs < min_size) & (cs >= 1e-3)
                cs_factors[cs_low] = min_size / cs[cs_low]
            
            Yt = Yt * cs_factors.unsqueeze(0)
        
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
        
        for iteration in range(1, iterations + 1):
            iter_start = time.time()
            
            # Aggregate to major cell types
            Ytk = torch.zeros(K, S, device=device)
            for k, ct in enumerate(cell_types):
                indices = self.ref['clusters'][ct]
                Ytk[k] = Yt[indices].sum(dim=0)
            
            rho_tk = Ytk.sum(dim=1)
            rho_tk = torch.clamp(rho_tk, min=1e-10)
            rho_t_ex = rho_tk[cluster_to_major]
            
            # Initialize gradient
            Dt = torch.zeros_like(Yt)
            
            # Abundance matching term
            ratio_error = 0.0
            if l_a > 0:
                # Jensen-Shannon divergence
                rho_avg = (rho_tk + r_sc) / 2
                log_rho = 0.5 * self._safe_log2(rho_tk / rho_avg)
                log_rsc = 0.5 * self._safe_log2(r_sc / rho_avg)
                
                ratio_error = (rho_tk * log_rho).sum() + (r_sc * log_rsc).sum()
                
                log_rho_ex = log_rho[cluster_to_major]
                d_ratio = l_a * log_rho_ex
                
                Dt += d_ratio.unsqueeze(1).expand_as(Yt)
            
            # Gene expression matching terms
            dcosine_st = 0.0
            dcosine_g = 0.0
            
            if (sparsity_coef < 1 and l_i > 0) or l_g > 0:
                # Predicted expression: Y^T X
                st_xt = Yt.T @ self.ref['X']
                
                if sparsity_coef < 1 and l_i > 0:
                    # Spot-wise cosine distance
                    st_xt_norms = torch.norm(st_xt, dim=1, keepdim=True)
                    st_xt_n = st_xt / (st_xt_norms + 1e-10)
                    
                    csi = (self.spatial['X'] * st_xt_n).sum(dim=1)
                    di = 1 - csi
                    di = torch.clamp(di, min=0)
                    
                    d_i_grad = 1.0 / (2 * torch.sqrt(di) + 1e-10)
                    di = torch.sqrt(di)
                    
                    dcosine_st = di.sum()
                    
                    st_de = l_i * (1 - sparsity_coef) * (
                        self.spatial['X'] - st_xt_n * csi.unsqueeze(1)
                    ) * d_i_grad.unsqueeze(1) / (st_xt_norms + 1e-10)
                    
                    Dt -= st_de @ self.ref['X'].T
                
                if l_g > 0:
                    # Gene-wise cosine distance
                    st_xg = F.normalize(self.spatial['X'], p=2, dim=0)
                    st_xt_gnorms = torch.norm(st_xt, dim=0, keepdim=True)
                    st_xt_gn = st_xt / (st_xt_gnorms + 1e-10)
                    
                    csg = (st_xt_gn * st_xg).sum(dim=0)
                    dg = 1 - csg
                    dg = torch.clamp(dg, min=0)
                    
                    dg_coefs = 1.0 / (2 * torch.sqrt(dg) + 1e-10) / (st_xt_gnorms + 1e-10)
                    dg = torch.sqrt(dg)
                    
                    dcosine_g = dg.sum()
                    
                    st_de_g = l_g * (st_xg - st_xt_gn * csg.unsqueeze(0)) * dg_coefs
                    Dt -= st_de_g @ self.ref['X'].T
            
            # Linear sparsity term
            dcosine_lin = 0.0
            if l_sp > 0:
                Dt += l_sp * linear_dcosine
                dcosine_lin = (Yt * linear_dcosine).sum()
            
            # Spatial coherence term
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
                        d_s += w * (Ytk[:, ii] * l_ii).sum()
                        Dtk[:, ii] += l_s * w * l_ii
                
                # Map back to subclusters
                for k, ct in enumerate(cell_types):
                    indices = self.ref['clusters'][ct]
                    Dt[indices] += Dtk[k].unsqueeze(0)
            
            # Frank-Wolfe step: find optimal direction
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
            
            gap = (Dt * (Yt - Yt_h)).sum()
            
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
            history['objective'].append(ft.item())
            history['upper_bound'].append(f_best.item())
            history['lower_bound'].append(lb.item() if lb != float('-inf') else None)
            history['gap'].append(rel_gap.item())
            history['time'].append(iter_time)
            
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
        self.solution = Y_best
        
        # Compute cell type weights
        self.weights = torch.zeros(S, K, device=device)
        for k, ct in enumerate(cell_types):
            indices = self.ref['clusters'][ct]
            self.weights[:, k] = Y_best[indices].sum(dim=0)
        
        self.history = history
        
        if verbose:
            print(f"\nOptimization complete. Final objective: {f_best:.4f}")
    
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
        
        weights = self.weights.cpu().numpy()
        
        if normalize:
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            weights = weights / row_sums
        
        return weights
    
    def get_cell_types(self) -> list:
        """Get list of cell type names."""
        return list(self.ref['clusters'].keys())
