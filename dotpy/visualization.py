"""
Visualization utilities for DOT results

Functions for plotting cell type abundances on spatial tissue.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, Tuple


def plot_spatial_weights(
    coords: np.ndarray,
    weights: np.ndarray,
    cell_types: Optional[list] = None,
    normalize: bool = True,
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    point_size: float = 1.0,
    cmap: str = 'magma',
    flip_y: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    background_color: str = '#E5E5E5',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """Plot spatial distribution of cell type weights."""
    if normalize:
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums

    n_ct = weights.shape[1]
    if cell_types is None:
        cell_types = [f"CT{i+1}" for i in range(n_ct)]

    coords_plot = coords.copy()
    if flip_y:
        coords_plot[:, 1] = -coords_plot[:, 1]

    nrows = int(np.ceil(n_ct / ncols))
    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    if n_ct == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = weights.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_ct):
        ax = axes[i]
        order = np.argsort(weights[:, i])
        scatter = ax.scatter(
            coords_plot[order, 0], coords_plot[order, 1],
            c=weights[order, i], s=point_size,
            cmap=cmap, norm=norm, rasterized=True
        )
        ax.set_aspect('equal')
        ax.set_facecolor(background_color)
        ax.set_title(cell_types[i], fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

    for i in range(n_ct, len(axes)):
        axes[i].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    return fig


def plot_cell_type_proportions(
    weights: np.ndarray,
    cell_types: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[list] = None,
    save_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """Plot overall cell type proportions across all spots."""
    n_ct = weights.shape[1]
    if cell_types is None:
        cell_types = [f"CT{i+1}" for i in range(n_ct)]

    proportions = weights.sum(axis=0) / weights.sum()
    order = np.argsort(proportions)[::-1]
    proportions = proportions[order]
    ct_sorted = [cell_types[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = [plt.cm.tab20(x) for x in np.linspace(0, 1, n_ct)]
        colors = [colors[i] for i in order]

    bars = ax.barh(range(n_ct), proportions, color=colors)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(ct_sorted, fontsize=10)
    ax.set_xlabel('Proportion', fontsize=12, fontweight='bold')
    ax.set_title('Cell Type Proportions', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, p in zip(bars, proportions):
        ax.text(p + 0.005, bar.get_y() + bar.get_height()/2,
                f'{p:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    return fig


def plot_optimization_history(
    history: dict,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """Plot DOT optimisation convergence history."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    iters = history['iteration']

    ax = axes[0]
    ax.plot(iters, history['objective'], 'b-', lw=2, label='Objective')
    ax.plot(iters, history['upper_bound'], 'r--', lw=1.5, label='Upper bound')
    lb = [x if x is not None else np.nan for x in history['lower_bound']]
    if not all(np.isnan(v) for v in lb):
        ax.plot(iters, lb, 'g--', lw=1.5, label='Lower bound')
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Objective', fontweight='bold')
    ax.set_title('Convergence', fontweight='bold')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(iters, history['gap'], 'b-', lw=2)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Relative gap', fontweight='bold')
    ax.set_title('Duality gap', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(iters, history['time'], 'b-', lw=2)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Time (s)', fontweight='bold')
    ax.set_title('Time per iteration', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    return fig