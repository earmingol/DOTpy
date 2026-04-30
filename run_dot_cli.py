#!/usr/bin/env python
# coding: utf-8

"""
DOT Analysis Pipeline - Command Line Interface

Run DOT deconvolution on spatial transcriptomics data with command-line arguments.

Usage:
    python run_dot_cli.py --ref reference.h5ad --spatial spatial.h5ad
    python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --output my_results
    python run_dot_cli.py --ref ref.h5ad --spatial spatial.h5ad --device cuda --mixed-precision
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from dotpy import DOT, setup_reference, setup_spatial
from dotpy.visualization import plot_spatial_weights, plot_optimization_history

try:
    from natsort import natsorted
except ImportError:
    # Fallback: plain sorted when natsort is not installed
    natsorted = sorted


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run DOT deconvolution on spatial transcriptomics data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--ref', '--reference', required=True,
                        help='Path to reference scRNA-seq h5ad file')
    parser.add_argument('--spatial', required=True,
                        help='Path to spatial transcriptomics h5ad file')

    # Column names
    parser.add_argument('--cell-type-key', default='cell_type',
                        help='Column in ref_adata.obs with cell type annotations')
    parser.add_argument('--sample-key', default=None,
                        help='Column in spatial_adata.obs with sample IDs (optional)')
    parser.add_argument('--lineage-key', default=None,
                        help='Column in ref_adata.obs with lineage annotations (optional)')
    parser.add_argument('--counts-layer', default='counts',
                        help='Layer in spatial_adata with raw counts (or "X" to use .X)')

    # Output
    parser.add_argument('--output', '-o', default='dot_results',
                        help='Output prefix for result files')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for result files')
    parser.add_argument('--save-combined', action='store_true',
                        help='Save combined results file at the end')

    # DOT parameters
    parser.add_argument('--subcluster-size', type=int, default=10,
                        help='Maximum number of subclusters per cell type')
    parser.add_argument('--max-genes', type=int, default=5000,
                        help='Maximum number of genes to use')
    parser.add_argument('--th-spatial', type=float, default=0.84,
                        help='Threshold on similarity of adjacent spots')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Batch size for GPU processing')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of optimization iterations')
    parser.add_argument('--mode', choices=['highres', 'lowres'], default='highres',
                        help='Resolution mode (highres for Xenium/MERFISH, lowres for Visium)')
    parser.add_argument('--ratios-weight', type=float, default=0.0,
                        help='Weight for matching reference cell-type abundances (0 disables; '
                             'try ~0.3 for lowres deconvolution)')

    # Device & performance
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto',
                        help='Compute device. "auto" selects CUDA when available.')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use float16 intermediates on GPU to reduce memory and speed up matmuls')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', default=None,
                        help='Directory for saving optimisation checkpoints (disabled by default)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='Save a checkpoint every N iterations')
    parser.add_argument('--resume-from', default=None,
                        help='Path to a checkpoint file to resume from')

    # Flags
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--no-h5ad', action='store_true',
                        help='Skip saving h5ad files per sample')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress messages')

    return parser.parse_args()


def _resolve_device(choice: str) -> str:
    """Return 'cuda' or 'cpu' from the user's --device flag."""
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return choice


def main():
    """Main analysis pipeline."""
    args = parse_args()

    device = _resolve_device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure figure directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = figures_dir

    print("=" * 70)
    print("DOT Analysis Pipeline")
    print("=" * 70)
    print(f"\nInput files:")
    print(f"  Reference:    {args.ref}")
    print(f"  Spatial:      {args.spatial}")
    print(f"\nDevice:           {device}")
    if device == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:            {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    print(f"Mixed precision:  {'ON' if args.mixed_precision else 'OFF'}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Figures directory: {figures_dir}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    if args.verbose:
        print("\n" + "=" * 70)
        print("1. Loading data")
        print("=" * 70)
    else:
        print("\n1. Loading data...")

    ref_adata = sc.read_h5ad(args.ref)
    spatial_adata = sc.read_h5ad(args.spatial)

    print(f"  Reference: {ref_adata.shape}")
    print(f"  Spatial: {spatial_adata.shape}")

    # Set counts to .X if needed
    if args.counts_layer != 'X':
        if args.counts_layer in spatial_adata.layers:
            if args.verbose:
                print(f"  Using counts from layer: {args.counts_layer}")
            spatial_adata.X = spatial_adata.layers[args.counts_layer].copy()
        else:
            print(f"  Warning: Layer '{args.counts_layer}' not found, using .X")

    # -------------------------------------------------------------------------
    # 2. Process reference
    # -------------------------------------------------------------------------
    if args.verbose:
        print("\n" + "=" * 70)
        print("2. Processing reference")
        print("=" * 70)
    else:
        print("\n2. Processing reference...")

    ref_processed = setup_reference(
        ref_adata,
        cell_type_key=args.cell_type_key,
        subcluster_size=args.subcluster_size,
        max_genes=args.max_genes,
        verbose=args.verbose
    )

    # -------------------------------------------------------------------------
    # 3. Get sample list
    # -------------------------------------------------------------------------
    if args.sample_key is not None and args.sample_key in spatial_adata.obs.columns:
        samples = spatial_adata.obs[args.sample_key].unique()
        print(f"\n3. Found {len(samples)} samples: {list(samples)}")
    else:
        samples = ['all']
        args.sample_key = None
        print("\n3. Processing as single sample")

    # -------------------------------------------------------------------------
    # 4. Process each sample and save results immediately
    # -------------------------------------------------------------------------
    if args.verbose:
        print("\n" + "=" * 70)
        print("4. Running DOT deconvolution")
        print("=" * 70)
    else:
        print("\n4. Running DOT deconvolution...")

    # Track lineage mapping dictionary
    lin_dict = None
    if args.lineage_key is not None and args.lineage_key in ref_adata.obs.columns:
        lin_dict = (ref_adata.obs[[args.cell_type_key, args.lineage_key]]
                    .groupby(args.cell_type_key)[args.lineage_key]
                    .agg(lambda x: x.value_counts().idxmax())
                    .to_dict()
                )

    # Initialize as string columns to avoid categorical issues during loop
    spatial_adata.obs['cell_type'] = ''
    if lin_dict is not None:
        spatial_adata.obs['lineage'] = ''

    # Process each sample
    for sample_id in samples:
        if args.verbose:
            print(f"\n{'=' * 70}")
            print(f"Processing: {sample_id}")
            print('=' * 70)
        else:
            print(f"  Processing sample: {sample_id}")

        # Subset data
        if args.sample_key is not None:
            sample_mask = spatial_adata.obs[args.sample_key] == sample_id
            sample_data = spatial_adata[sample_mask].copy()
            sample_indices = spatial_adata.obs.index[sample_mask]
        else:
            sample_data = spatial_adata.copy()
            sample_indices = spatial_adata.obs.index

        if args.verbose:
            print(f"Cells in sample: {sample_data.shape[0]}")

        # Process spatial
        spatial_processed = setup_spatial(
            sample_data,
            spatial_key='spatial',
            th_spatial=args.th_spatial,
            radius='auto',
            verbose=args.verbose
        )

        # Run DOT
        dot = DOT(
            spatial_processed,
            ref_processed,
            batch_size=args.batch_size,
            device=device
        )

        # Per-sample checkpoint dir (when multiple samples)
        ckpt_dir = None
        if args.checkpoint_dir is not None:
            ckpt_dir = str(Path(args.checkpoint_dir) / str(sample_id))

        dot.fit(
            mode=args.mode,
            ratios_weight=args.ratios_weight,
            iterations=args.iterations,
            verbose=args.verbose,
            use_mixed_precision=args.mixed_precision,
            checkpoint_dir=ckpt_dir,
            checkpoint_freq=args.checkpoint_freq,
            resume_from=args.resume_from,
        )

        # Get results
        weights = dot.get_weights(normalize=True)
        cell_types = dot.get_cell_types()

        # Create weights DataFrame
        weights_df = pd.DataFrame(
            weights,
            index=sample_indices,
            columns=cell_types
        )

        # Assign cell types
        sample_data.obs['cell_type'] = weights_df.idxmax(axis=1)

        # Map lineages if requested
        if lin_dict is not None:
            sample_data.obs['lineage'] = sample_data.obs['cell_type'].map(lin_dict)

        # =====================================================================
        # SAVE RESULTS IMMEDIATELY
        # =====================================================================
        sample_suffix = f"_{sample_id}".replace(' ', '_').replace('/', '_')

        # Save weights
        weights_path = output_dir / f"{args.output}{sample_suffix}_weights.csv"
        weights_df.to_csv(weights_path)
        if args.verbose:
            print(f"  ✓ Saved weights: {weights_path}")

        # Save annotations
        annotation_cols = ['cell_type']
        if 'lineage' in sample_data.obs.columns:
            annotation_cols.append('lineage')

        annotations_df = sample_data.obs[annotation_cols]
        annotations_path = output_dir / f"{args.output}{sample_suffix}_annotations.csv"
        annotations_df.to_csv(annotations_path)
        if args.verbose:
            print(f"  ✓ Saved annotations: {annotations_path}")

        # Save h5ad if requested
        if not args.no_h5ad:
            h5ad_path = output_dir / f"{args.output}{sample_suffix}.h5ad"
            sample_data.write(h5ad_path)
            if args.verbose:
                print(f"  ✓ Saved h5ad: {h5ad_path}")

        # Generate plots if requested
        if not args.no_plots:
            try:
                sq.pl.spatial_scatter(
                    sample_data,
                    library_id="spatial",
                    shape=None,
                    color='cell_type',
                    na_color='whitesmoke',
                    wspace=0.4,
                    legend_loc="right margin",
                    figsize=(15, 15),
                    save=f'_{args.output}{sample_suffix}_cell_types.png'
                )
                if args.verbose:
                    print(f"  ✓ Saved cell type plot")

                if 'lineage' in sample_data.obs.columns:
                    sq.pl.spatial_scatter(
                        sample_data,
                        library_id="spatial",
                        shape=None,
                        color='lineage',
                        na_color='whitesmoke',
                        wspace=0.4,
                        legend_loc="right margin",
                        figsize=(15, 15),
                        save=f'_{args.output}{sample_suffix}_lineages.png'
                    )
                    if args.verbose:
                        print(f"  ✓ Saved lineage plot")

                weights_plot_path = figures_dir / f"{args.output}{sample_suffix}_weights.png"
                fig = plot_spatial_weights(
                    coords=sample_data.obsm['spatial'],
                    weights=weights,
                    cell_types=cell_types,
                    save_path=str(weights_plot_path),
                )
                plt.close(fig)
                if args.verbose:
                    print(f"  ✓ Saved weights plot: {weights_plot_path}")

                history_plot_path = figures_dir / f"{args.output}{sample_suffix}_convergence.png"
                fig = plot_optimization_history(
                    dot.history,
                    save_path=str(history_plot_path),
                )
                plt.close(fig)
                if args.verbose:
                    print(f"  ✓ Saved convergence plot: {history_plot_path}")

            except Exception as e:
                print(f"  Warning: Could not generate plots: {e}")

        if not args.verbose:
            print(f"    ✓ Completed and saved")

        # Update spatial_adata obs with results
        spatial_adata.obs.loc[sample_indices, 'cell_type'] = sample_data.obs['cell_type'].astype(str)
        if 'lineage' in sample_data.obs.columns:
            spatial_adata.obs.loc[sample_indices, 'lineage'] = sample_data.obs['lineage'].astype(str)

        # Free GPU memory between samples
        del dot, spatial_processed
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Convert to categorical with natural sorting
    # -------------------------------------------------------------------------
    if args.verbose:
        print("\nConverting to categorical with natural sorting...")

    all_cell_types = natsorted(spatial_adata.obs['cell_type'].unique())
    spatial_adata.obs['cell_type'] = pd.Categorical(
        spatial_adata.obs['cell_type'],
        categories=all_cell_types,
        ordered=True
    )

    if 'lineage' in spatial_adata.obs.columns:
        all_lineages = natsorted(spatial_adata.obs['lineage'].unique())
        spatial_adata.obs['lineage'] = pd.Categorical(
            spatial_adata.obs['lineage'],
            categories=all_lineages,
            ordered=True
        )

    # -------------------------------------------------------------------------
    # 5. Optionally save combined results
    # -------------------------------------------------------------------------
    if args.save_combined:
        if args.verbose:
            print("\n" + "=" * 70)
            print("5. Saving combined results")
            print("=" * 70)
        else:
            print("\n5. Saving combined results...")

        # Combined weights
        weights_path = output_dir / f"{args.output}_combined_weights.csv"

        all_weights = []
        for sample_id in samples:
            sample_suffix = f"_{sample_id}".replace(' ', '_').replace('/', '_')
            sample_weights_path = output_dir / f"{args.output}{sample_suffix}_weights.csv"
            sample_weights = pd.read_csv(sample_weights_path, index_col=0)
            all_weights.append(sample_weights)

        combined_weights = pd.concat(all_weights)
        combined_weights = combined_weights.reindex(spatial_adata.obs.index)
        combined_weights.to_csv(weights_path)
        print(f"  ✓ Saved combined weights: {weights_path}")

        # Combined annotations
        annotation_cols = ['cell_type']
        if 'lineage' in spatial_adata.obs.columns:
            annotation_cols.append('lineage')

        annotations_df = spatial_adata.obs[annotation_cols]
        annotations_path = output_dir / f"{args.output}_combined_annotations.csv"
        annotations_df.to_csv(annotations_path)
        print(f"  ✓ Saved combined annotations: {annotations_path}")

        if not args.no_h5ad:
            h5ad_path = output_dir / f"{args.output}_combined.h5ad"
            spatial_adata.write(h5ad_path)
            print(f"  ✓ Saved combined h5ad: {h5ad_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nProcessed {len(samples)} sample(s) on {device.upper()}")
    print(f"\nPer-sample results saved to: {output_dir}")
    print(f"  Pattern: {args.output}_<sample>_{{weights,annotations}}.csv")
    if not args.no_plots:
        print(f"\nFigures saved to: {figures_dir}")
    if args.save_combined:
        print(f"\nCombined results also saved")

    return spatial_adata


if __name__ == '__main__':
    spatial_adata = main()