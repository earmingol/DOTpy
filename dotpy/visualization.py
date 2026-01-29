"""
Visualization utilities for DOT results

Functions for plotting cell type abundances on spatial tissue.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, Union, Tuple
import warnings


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
    """
    Plot spatial distribution of cell type weights.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (spots × 2)
    weights : np.ndarray
        Cell type weights (spots × cell_types)
    cell_types : list, optional
        Names of cell types
    normalize : bool
        Normalize weights to proportions per spot
    ncols : int
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size (width, height)
    point_size : float
        Size of scatter points
    cmap : str
        Matplotlib colormap name
    flip_y : bool
        Flip y-axis (common for histology images)
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    background_color : str
        Background color for plots
    title : str, optional
        Overall figure title
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Normalize weights if requested
    if normalize:
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums
    
    n_cell_types = weights.shape[1]
    
    if cell_types is None:
        cell_types = [f"CT{i+1}" for i in range(n_cell_types)]
    
    # Flip y coordinates if requested
    coords_plot = coords.copy()
    if flip_y:
        coords_plot[:, 1] = -coords_plot[:, 1]
    
    # Determine grid layout
    nrows = int(np.ceil(n_cell_types / ncols))
    
    if figsize is None:
        figsize = (ncols * 3, nrows * 3)
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    
    if n_cell_types == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Determine color scale
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = weights.max()
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each cell type
    for i in range(n_cell_types):
        ax = axes[i]
        
        # Sort by weight so high values are on top
        sort_idx = np.argsort(weights[:, i])
        coords_sorted = coords_plot[sort_idx]
        weights_sorted = weights[sort_idx, i]
        
        scatter = ax.scatter(
            coords_sorted[:, 0],
            coords_sorted[:, 1],
            c=weights_sorted,
            s=point_size,
            cmap=cmap,
            norm=norm,
            rasterized=True
        )
        
        ax.set_aspect('equal')
        ax.set_facecolor(background_color)
        ax.set_title(cell_types[i], fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for i in range(n_cell_types, len(axes)):
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
    """
    Plot overall cell type proportions across all spots.
    
    Parameters
    ----------
    weights : np.ndarray
        Cell type weights (spots × cell_types)
    cell_types : list, optional
        Names of cell types
    figsize : tuple
        Figure size
    colors : list, optional
        Colors for each cell type
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    n_cell_types = weights.shape[1]
    
    if cell_types is None:
        cell_types = [f"CT{i+1}" for i in range(n_cell_types)]
    
    # Compute mean proportions
    proportions = weights.sum(axis=0) / weights.sum()
    
    # Sort by proportion
    sort_idx = np.argsort(proportions)[::-1]
    proportions = proportions[sort_idx]
    cell_types_sorted = [cell_types[i] for i in sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, n_cell_types))
        colors = [colors[i] for i in sort_idx]
    
    bars = ax.barh(range(n_cell_types), proportions, color=colors)
    
    ax.set_yticks(range(n_cell_types))
    ax.set_yticklabels(cell_types_sorted, fontsize=10)
    ax.set_xlabel('Proportion', fontsize=12, fontweight='bold')
    ax.set_title('Cell Type Proportions', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (bar, prop) in enumerate(zip(bars, proportions)):
        ax.text(prop + 0.005, i, f'{prop:.3f}', 
                va='center', fontsize=9)
    
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
    """
    Plot DOT optimization history.
    
    Parameters
    ----------
    history : dict
        Optimization history from DOT.history
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    iterations = history['iteration']
    
    # Objective
    ax = axes[0]
    ax.plot(iterations, history['objective'], 'b-', linewidth=2, label='Objective')
    ax.plot(iterations, history['upper_bound'], 'r--', linewidth=1.5, label='Upper bound')
    if history['lower_bound'][0] is not None:
        lb = [x if x is not None else np.nan for x in history['lower_bound']]
        ax.plot(iterations, lb, 'g--', linewidth=1.5, label='Lower bound')
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Objective value', fontweight='bold')
    ax.set_title('Convergence', fontweight='bold')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    
    # Gap
    ax = axes[1]
    ax.semilogy(iterations, history['gap'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Relative gap', fontweight='bold')
    ax.set_title('Duality gap', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    
    # Time per iteration
    ax = axes[2]
    ax.plot(iterations, history['time'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Time (s)', fontweight='bold')
    ax.set_title('Time per iteration', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig