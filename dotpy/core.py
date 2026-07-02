import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict
from scipy.sparse import issparse
from scipy import linalg as sp_linalg
import time
import pickle
from pathlib import Path


class DOT:
    """Deconvolution by Optimal Transport – GPU-optimised batched solver."""

    def __init__(
        self,
        spatial: Dict,
        ref: Dict,
        ls_solution: bool = True,
        batch_size: int = 500,
        device: Optional[str] = None,
    ):
        # --- Gene alignment ---
        spatial_genes = np.asarray(spatial['genes'])
        ref_genes = np.asarray(ref['genes'])
        common_genes = np.intersect1d(spatial_genes, ref_genes)
        if len(common_genes) == 0:
            raise ValueError("No common genes found between spatial and reference data")

        # Index both matrices in the same explicit order. Filtering each array with
        # np.isin() preserves its original order, which may differ after DE ranking.
        sp_lookup = {gene: i for i, gene in enumerate(spatial_genes)}
        rf_lookup = {gene: i for i, gene in enumerate(ref_genes)}
        sp_idx = np.fromiter((sp_lookup[g] for g in common_genes), dtype=np.int64)
        rf_idx = np.fromiter((rf_lookup[g] for g in common_genes), dtype=np.int64)

        X_sp = spatial['X_sparse'][:, sp_idx] if issparse(spatial['X_sparse']) \
            else spatial['X_sparse'][:, sp_idx]
        X_rf = ref['X_sparse'][:, rf_idx] if issparse(ref['X_sparse']) \
            else ref['X_sparse'][:, rf_idx]

        self.spatial = {
            'X_sparse': X_sp,
            'coords': spatial['coords'],
            'genes': common_genes,
            'device': device or 'cpu',
        }
        if 'pairs' in spatial:
            self.spatial['pairs'] = spatial['pairs']

        self.ref = {
            'X_sparse': X_rf,
            'clusters': ref['clusters'],
            'ratios': ref['ratios'],
            'genes': common_genes,
            'device': device or 'cpu',
        }

        self.batch_size = batch_size
        self.solution = None
        self.weights = None
        self.history = None

        if ls_solution:
            self.solution = self._ls_solution()

    # ------------------------------------------------------------------
    # Least-squares initialisation
    # ------------------------------------------------------------------
    def _ls_solution(self, lambda_ridge: float = 100.0) -> np.ndarray:
        """Ridge-regularised LS init.  Exploits sparsity when possible."""
        X_ref = self.ref['X_sparse']
        X_sp = self.spatial['X_sparse']

        if issparse(X_ref):
            X_ref_d = X_ref.toarray().astype(np.float32)
        else:
            X_ref_d = np.asarray(X_ref, dtype=np.float32)

        if issparse(X_sp):
            X_sp_d = X_sp.toarray().astype(np.float32)
        else:
            X_sp_d = np.asarray(X_sp, dtype=np.float32)

        C = X_ref_d.shape[0]

        # (Xref Xref^T + λI) is small (C×C), solve directly
        XtX = X_ref_d @ X_ref_d.T
        XtX[np.diag_indices(C)] += lambda_ridge

        XtY = X_ref_d @ X_sp_d.T  # C × S

        # Use scipy's optimised symmetric positive-definite solver
        solution = sp_linalg.solve(XtX, XtY, assume_a='pos', overwrite_a=True)
        np.maximum(solution, 0, out=solution)
        return solution

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------
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
        resume_from: Optional[str] = None,
        use_mixed_precision: bool = False,
    ) -> 'DOT':
        """
        Run DOT optimisation.

        Parameters
        ----------
        mode : ``'highres'`` or ``'lowres'``
        ratios_weight : float
            Weight for matching reference cell-type abundances.
        max_spot_size : int
            Max cells per spot (lowres mode).
        iterations : int
            Frank-Wolfe iterations.
        gap_threshold : float
            Relative duality gap for convergence.
        verbose : bool
        checkpoint_dir / checkpoint_freq : checkpointing.
        resume_from : str, optional
            Path to a previous checkpoint.
        use_mixed_precision : bool
            Use float16 intermediates on GPU (saves memory).
        """
        if mode == 'highres':
            sparsity_coef, max_size = 0.6, 1
        elif mode == 'lowres':
            sparsity_coef, max_size = 0.4, max_spot_size
        else:
            raise ValueError("mode must be 'highres' or 'lowres'")

        start_iter = 1
        if resume_from is not None:
            start_iter = self._load_checkpoint(resume_from, verbose)

        self._run_optimisation(
            ratios_weight=ratios_weight,
            sparsity_coef=sparsity_coef,
            max_size=max_size,
            min_size=1,
            iterations=iterations,
            gap_threshold=gap_threshold,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            start_iteration=start_iter,
            use_mixed_precision=use_mixed_precision,
        )
        return self

    # ------------------------------------------------------------------
    # Core optimisation loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_optimisation(
        self,
        ratios_weight, sparsity_coef, max_size, min_size,
        iterations, gap_threshold, verbose,
        checkpoint_dir, checkpoint_freq, start_iteration,
        use_mixed_precision=False,
    ):
        device_str = self.ref['device']
        use_gpu = device_str == 'cuda' and torch.cuda.is_available()
        device = torch.device(device_str if use_gpu else 'cpu')

        # Compute dtype – optionally float16 for intermediates
        compute_dtype = torch.float32
        if use_mixed_precision and use_gpu:
            compute_dtype = torch.float16

        # ============================================================
        # 1. Prepare data on CPU
        # ============================================================
        X_ref_np = self.ref['X_sparse'].toarray().astype(np.float32) \
            if issparse(self.ref['X_sparse']) else np.asarray(self.ref['X_sparse'], dtype=np.float32)
        X_sp_np = self.spatial['X_sparse'].toarray().astype(np.float32) \
            if issparse(self.spatial['X_sparse']) else np.asarray(self.spatial['X_sparse'], dtype=np.float32)

        S, G = X_sp_np.shape
        C = X_ref_np.shape[0]

        cell_types = list(self.ref['clusters'].keys())
        K = len(cell_types)

        # Cluster → major type mapping (vectorised)
        cluster_to_major = np.zeros(C, dtype=np.int64)
        cluster_indices_list = []            # list of np arrays per major type
        for k, ct in enumerate(cell_types):
            idx = np.asarray(self.ref['clusters'][ct])
            cluster_to_major[idx] = k
            cluster_indices_list.append(idx)

        sc_ratios = np.array([self.ref['ratios'][ct] for ct in cell_types], dtype=np.float32)
        sc_ratios /= sc_ratios.sum()

        r_st = np.full(S, 0.9 * min_size + 0.1 * max_size, dtype=np.float32)
        n_st = r_st.sum()
        r_sc = sc_ratios * n_st
        r_sc_ex = r_sc[cluster_to_major]

        # Loss weights
        inner = [1.0, 0.25 if max_size > 1 else 1.0, 0.0, 0.01]
        l_a = ratios_weight / max_size
        l_g = inner[0] * S / G
        l_i = inner[1]
        l_sp = l_i * sparsity_coef / max_size

        has_pairs = 'pairs' in self.spatial and self.spatial['pairs'] is not None
        if has_pairs:
            l_s = inner[3] * S / (max_size * len(self.spatial['pairs']['i']))
            pairs_i_np = self.spatial['pairs']['i'].astype(np.int64)
            pairs_j_np = self.spatial['pairs']['j'].astype(np.int64)
            pairs_w_np = self.spatial['pairs']['w'].astype(np.float32)
        else:
            l_s = 0.0

        # ============================================================
        # 2. Move to device ONCE
        # ============================================================
        X_ref = torch.from_numpy(X_ref_np).to(device)      # C × G
        X_sp = torch.from_numpy(X_sp_np).to(device)         # S × G
        X_ref_norm = F.normalize(X_ref, p=2, dim=1)         # C × G  (L2-normed rows)
        # R DOT computes ST_Xn <- normalize(ST_X) once and uses it in both
        # spot-wise cosine terms.
        X_sp_row_norm = F.normalize(X_sp, p=2, dim=1)       # S × G

        c2m = torch.from_numpy(cluster_to_major).to(device)
        r_sc_t = torch.from_numpy(r_sc).to(device)
        sc_ratios_t = torch.from_numpy(sc_ratios).to(device)
        r_st_t = torch.from_numpy(r_st).to(device)

        # Pre-build scatter indices for cluster→major aggregation
        # cluster_scatter[c] = k  (same as c2m, but we keep both for clarity)
        # For fast scatter_add we need indices repeated for all S columns –
        # but that's too large; instead we'll use a loop over K major types
        # with pre-built index tensors (each is small).
        major_idx_tensors = []
        for k in range(K):
            major_idx_tensors.append(
                torch.from_numpy(cluster_indices_list[k]).to(device)
            )

        if has_pairs:
            p_i = torch.from_numpy(pairs_i_np).to(device)
            p_j = torch.from_numpy(pairs_j_np).to(device)
            p_w = torch.from_numpy(pairs_w_np).to(device) * 0.5 / np.log(2)

        del X_ref_np, X_sp_np  # free CPU copies

        # ============================================================
        # 3. Initialise solution Yt  (C × S) on device
        # ============================================================
        Yt = None
        if self.solution is not None:
            Yt_cand = torch.from_numpy(self.solution.astype(np.float32)).to(device)
            if Yt_cand.shape == (C, S):
                Yt_cand.clamp_(min=0)
                cs = Yt_cand.sum(dim=0)
                small = cs < 1e-3
                if small.any():
                    Yt_cand[:, small] = 1.0 / C

                cs_factors = torch.ones(S, device=device)
                if sparsity_coef > 0.5:
                    cs_factors[~small] = 1.0 / cs[~small]
                else:
                    cs_high = cs > max_size
                    cs_factors[cs_high] = max_size / cs[cs_high]
                    cs_low = (cs < min_size) & ~small
                    cs_factors[cs_low] = min_size / cs[cs_low]

                Yt_cand.mul_(cs_factors.unsqueeze(0))
                if (Yt_cand == 0).any():
                    cs_weight = 0.99
                    Yt_cand = Yt_cand * cs_weight + (1.0 - cs_weight) / C
                Yt = Yt_cand

        if Yt is None:
            initial = torch.zeros(C, device=device)
            for k, ct in enumerate(cell_types):
                idx_t = major_idx_tensors[k]
                initial[idx_t] = sc_ratios_t[k] / len(idx_t)
            Yt = initial.unsqueeze(1) * r_st_t.unsqueeze(0)

            mix_weight = 0.1
            Yt = Yt * mix_weight

            linear_dcosine = 1 - X_ref_norm @ X_sp_row_norm.T
            c_min = linear_dcosine.argmin(dim=0)
            s_idx = torch.arange(S, device=device)
            Yt[c_min, s_idx] += (1 - mix_weight) * r_st_t

            del linear_dcosine

        # ============================================================
        # 4. Optimisation loop
        # ============================================================
        f_best = float('inf')
        lb = float('-inf')
        Y_best = None

        history = {k: [] for k in ['iteration', 'objective', 'upper_bound',
                                    'lower_bound', 'gap', 'time']}

        lg2 = np.log(2)
        batch = self.batch_size
        n_batches = int(np.ceil(S / batch))

        if verbose:
            dev_name = "GPU" if use_gpu else "CPU"
            print(f"Optimising on {dev_name} | S={S} C={C} G={G} K={K}")
            print(f"  batches={n_batches} × {batch} spots"
                  f" | mixed_prec={'ON' if compute_dtype==torch.float16 else 'OFF'}")

        for iteration in range(start_iteration, iterations + 1):
            t0 = time.time()

            # ---- Aggregate Yt → Ytk  (K × S) ----
            Ytk = torch.zeros(K, S, device=device, dtype=torch.float32)
            for k in range(K):
                idx_t = major_idx_tensors[k]
                Ytk[k] = Yt[idx_t].sum(dim=0)

            rho_tk = Ytk.sum(dim=1).clamp(min=1e-10)       # K
            rho_t_ex = rho_tk[c2m]                           # C

            Dt = torch.zeros_like(Yt)                        # C × S  (gradient)

            # ============ ABUNDANCE MATCHING ============
            ratio_err = 0.0
            if l_a > 0:
                rho_avg = (rho_tk + r_sc_t) * 0.5
                log_rho = 0.5 * _safe_log2(rho_tk / rho_avg)
                log_rsc = 0.5 * _safe_log2(r_sc_t / rho_avg)
                ratio_err = (rho_tk * log_rho + r_sc_t * log_rsc).sum().item()
                d_ratio = l_a * log_rho[c2m]                # C
                Dt.add_(d_ratio.unsqueeze(1))

            # ============ SPOT-WISE COSINE  (batched) ============
            dcosine_st = 0.0
            dcosine_lin = 0.0

            need_spot = (sparsity_coef < 1 and l_i > 0) or l_sp > 0
            if need_spot:
                for b in range(n_batches):
                    s0 = b * batch
                    s1 = min(s0 + batch, S)
                    Yt_b = Yt[:, s0:s1]                       # C × b
                    Xsp_b = X_sp[s0:s1]                        # b × G
                    Xsp_b_n = X_sp_row_norm[s0:s1]             # b × G

                    # predicted expression  b × G
                    if compute_dtype == torch.float16:
                        st_xt = Yt_b.T.half() @ X_ref.half()
                        st_xt = st_xt.float()
                    else:
                        st_xt = Yt_b.T @ X_ref

                    # -- spot-wise cosine --
                    if sparsity_coef < 1 and l_i > 0:
                        norms = st_xt.norm(dim=1, keepdim=True).clamp(min=1e-10)
                        st_xt_n = st_xt / norms

                        csi = (Xsp_b_n * st_xt_n).sum(dim=1)
                        di = (1 - csi).clamp(min=0)
                        d_i_grad = _sqrt_env_grad(di)
                        di_sqrt = _sqrt_env(di)
                        dcosine_st += di_sqrt.sum().item()

                        coef = l_i * (1 - sparsity_coef)
                        st_de = coef * (Xsp_b_n - st_xt_n * csi.unsqueeze(1)) \
                            * d_i_grad.unsqueeze(1) / norms

                        if compute_dtype == torch.float16:
                            Dt[:, s0:s1] -= (st_de.half() @ X_ref.half().T).float().T
                        else:
                            Dt[:, s0:s1] -= (st_de @ X_ref.T).T

                    # -- linear sparsity --
                    if l_sp > 0:
                        if compute_dtype == torch.float16:
                            lin_d = (1 - X_ref_norm.half() @ Xsp_b_n.half().T).float()
                        else:
                            lin_d = 1 - X_ref_norm @ Xsp_b_n.T
                        lin_d.clamp_(min=0).sqrt_()
                        Dt[:, s0:s1].add_(lin_d, alpha=l_sp)
                        dcosine_lin += (Yt_b * lin_d).sum().item()

            # ============ GENE-WISE COSINE  (chunked) ============
            dcosine_g = 0.0
            if l_g > 0:
                # Full predicted expression  S × G  (may be large)
                # Chunk by genes to limit memory when G is huge
                GENE_CHUNK = max(256, G)  # process all at once if feasible
                st_xt_full = Yt.T @ X_ref                      # S × G

                st_gnorms = st_xt_full.norm(dim=0, keepdim=True).clamp(min=1e-10)
                st_gn = st_xt_full / st_gnorms                  # S × G

                X_sp_col_norm = F.normalize(X_sp, p=2, dim=0)   # S × G

                csg = (st_gn * X_sp_col_norm).sum(dim=0)        # G
                dg = (1 - csg).clamp(min=0)
                dg_coefs = _sqrt_env_grad(dg) / st_gnorms.squeeze(0)
                dg = _sqrt_env(dg)
                dcosine_g = dg.sum().item()

                # gradient  S × G
                st_de_g = l_g * (X_sp_col_norm - st_gn * csg.unsqueeze(0)) * dg_coefs.unsqueeze(0)
                Dt -= (st_de_g @ X_ref.T).T

                del st_xt_full, st_gn, st_de_g  # free

            # ============ SPATIAL COHERENCE  (vectorised) ============
            d_s = 0.0
            if l_s > 0 and has_pairs:
                # Ytk is K × S
                # Vectorised over all pairs at once
                n_pairs = p_i.shape[0]

                # Batch pairs if very many to limit memory
                PAIR_BATCH = min(n_pairs, 200_000)
                Dtk = torch.zeros_like(Ytk)

                for pb_start in range(0, n_pairs, PAIR_BATCH):
                    pb_end = min(pb_start + PAIR_BATCH, n_pairs)
                    bi = p_i[pb_start:pb_end]
                    bj = p_j[pb_start:pb_end]
                    bw = p_w[pb_start:pb_end]              # already *= 0.5/ln2

                    Yi = Ytk[:, bi]                        # K × batch_pairs
                    Yj = Ytk[:, bj]                        # K × batch_pairs
                    Ym = 0.5 * (Yi + Yj)                   # K × batch_pairs

                    log_Yi = _safe_log2(Yi / (Ym + 1e-10))
                    log_Yj = _safe_log2(Yj / (Ym + 1e-10))

                    # Weighted JSD contributions
                    w_exp = bw.unsqueeze(0)                # 1 × batch_pairs
                    d_s += (w_exp * (Yi * log_Yi + Yj * log_Yj)).sum().item()

                    grad_i = (l_s * w_exp) * log_Yi        # K × batch_pairs
                    grad_j = (l_s * w_exp) * log_Yj

                    # Scatter-add gradients back (vectorised)
                    Dtk.scatter_add_(1, bi.unsqueeze(0).expand(K, -1), grad_i)
                    Dtk.scatter_add_(1, bj.unsqueeze(0).expand(K, -1), grad_j)

                # Map Dtk (K×S) → Dt (C×S)
                for k in range(K):
                    idx_t = major_idx_tensors[k]
                    Dt[idx_t] += Dtk[k].unsqueeze(0)

            # ============ FRANK-WOLFE STEP  (vectorised) ============
            # argmin over C for each of S columns
            kk = torch.argmin(Dt, dim=0)                    # S
            min_vals = Dt[kk, torch.arange(S, device=device)]

            # Build Yt_h  (C × S) – sparse in practice
            Yt_h = torch.zeros_like(Yt)
            fill_val = torch.where(min_vals < 0,
                                   torch.tensor(float(max_size), device=device),
                                   torch.tensor(float(min_size), device=device))
            Yt_h[kk, torch.arange(S, device=device)] = fill_val

            # ---- Objective + gap ----
            ft = (l_i * (1 - sparsity_coef) * dcosine_st
                  + l_sp * dcosine_lin
                  + l_g * dcosine_g
                  + l_s * d_s
                  + l_a * ratio_err)

            gap = (Dt * (Yt - Yt_h)).sum().item()

            if ft < f_best:
                f_best = ft
                Y_best = Yt.clone()

            lb = max(lb, ft - gap)
            rel_gap = gap / abs(f_best) if abs(f_best) > 1e-10 else gap

            step = min(0.99, 2.0 / (iteration + 1))
            elapsed = time.time() - t0

            if verbose and (iteration % 10 == 1 or iteration == iterations):
                extras = f", GPU mem={torch.cuda.max_memory_allocated()/1024**2:.0f}MB" \
                    if use_gpu else ""
                print(f"Iter {iteration:3d}: obj={ft:8.4f}  gap={rel_gap:6.4f}  "
                      f"t={elapsed:5.2f}s{extras}")

            history['iteration'].append(iteration)
            history['objective'].append(ft)
            history['upper_bound'].append(f_best)
            history['lower_bound'].append(lb if lb != float('-inf') else None)
            history['gap'].append(rel_gap)
            history['time'].append(elapsed)

            # Checkpoint
            if checkpoint_dir and iteration % checkpoint_freq == 0:
                self._save_checkpoint(
                    checkpoint_dir, iteration,
                    Yt.cpu(), Y_best.cpu() if Y_best is not None else None,
                    f_best, lb, history, verbose
                )

            # Convergence
            converged = rel_gap <= gap_threshold and iteration >= 10
            if converged:
                # R: continue if best-objective dropped > gap_threshold over last ~5 iters
                look_back_iter = max(2, iteration - 5)
                look_back_idx = look_back_iter - start_iteration
                still_improving = False
                if 0 <= look_back_idx < len(history['upper_bound']):
                    f_old = history['upper_bound'][look_back_idx]
                    if f_old is not None and abs(f_best) > 1e-10:
                        if (f_old - f_best) / abs(f_best) >= gap_threshold:
                            still_improving = True
                if still_improving:
                    lb = float('-inf')
                    if verbose:
                        print(f"Iter {iteration}: resetting LB, still improving")
                else:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            if step <= 1e-5:
                if verbose:
                    print(f"Step size too small at iteration {iteration}")
                break

            # Update  (in-place lerp)
            Yt.lerp_(Yt_h, step)

        # ============================================================
        # 5. Store results
        # ============================================================
        self.solution = Y_best.cpu().numpy()

        weights = np.zeros((S, K), dtype=np.float32)
        for k in range(K):
            idx_t = major_idx_tensors[k]
            weights[:, k] = Y_best[idx_t].sum(dim=0).cpu().numpy()

        self.weights = weights
        self.history = history

        if verbose:
            print(f"\nDone. Final obj: {f_best:.4f}")
            if use_gpu:
                print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, ckpt_dir, iteration, Yt, Y_best, f_best, lb, history, verbose):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        path = Path(ckpt_dir) / f"checkpoint_iter_{iteration}.pkl"
        ckpt = {
            'iteration': iteration,
            'Yt': Yt.numpy() if torch.is_tensor(Yt) else Yt,
            'Y_best': Y_best.numpy() if Y_best is not None and torch.is_tensor(Y_best) else Y_best,
            'f_best': f_best, 'lb': lb,
            'history': history,
            'solution': self.solution,
        }
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)
        if verbose:
            print(f"  checkpoint → {path}")

    def _load_checkpoint(self, path, verbose):
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        self.solution = ckpt['Y_best']
        self.history = ckpt['history']
        if verbose:
            print(f"Resumed from {path} (iter {ckpt['iteration']})")
        return ckpt['iteration'] + 1

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def get_weights(self, normalize: bool = True) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Not fitted yet – call fit() first.")
        w = self.weights.copy()
        if normalize:
            rs = w.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            w /= rs
        return w

    def get_cell_types(self) -> list:
        return list(self.ref['clusters'].keys())


# ======================================================================
# Module-level helpers
# ======================================================================

def _safe_log2(x: torch.Tensor) -> torch.Tensor:
    """log2 that maps 0 → 0 and clips -inf."""
    out = torch.log2(x.clamp(min=1e-10))
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=-20.0)
    return out


_ENV_MIN = 1e-2
_ENV_SLOPE = 0.25 / _ENV_MIN
_ENV_THRESHOLD = 4 * _ENV_MIN ** 2


def _sqrt_env(v: torch.Tensor) -> torch.Tensor:
    """sqrt clamped near zero with linear envelope (matches R sqrt_env)."""
    return torch.where(v < _ENV_THRESHOLD, _ENV_SLOPE * v + _ENV_MIN, v.sqrt())


def _sqrt_env_grad(v: torch.Tensor) -> torch.Tensor:
    """Gradient of sqrt_env (matches R sqrt_env_grad)."""
    return torch.where(
        v < _ENV_THRESHOLD,
        torch.full_like(v, _ENV_SLOPE),
        0.5 / (v.sqrt() + 1e-10),
    )