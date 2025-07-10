#!/usr/bin/env python3
"""
CCS Training and Steering Pipeline - All-to-All Layer Analysis with COMPLETE Integration
========================================================================================

This script runs the complete pipeline for EVERY layer as a steering layer:
1. Load model and data
2. Extract representations
3. Train vanilla CCS on all layers
4. For each layer as "steering layer":
   - Apply steering using that layer
   - Run comprehensive analysis with ALL functions
   - Save results in layer-specific subdirectories

Each layer gets treated as the "best layer" and receives FULL analysis:
- Enhanced steering analysis with component separation
- ALL functions from steering_analysis.py
- ALL functions from steering_analysis1.py
- ALL functions from format_results.py
- Comprehensive comparison analysis
- Layer-wise PCA comparisons
- Component matrix analysis
- Best separation plots

Output structure:
pythia-1b_1B_timestamp/
├── layer_0/
│   ├── plots/
│   ├── logs/
│   └── results/
├── layer_1/
│   ├── plots/
│   ├── logs/
│   └── results/
...

Changed: Added ALL analysis functions from all modules
"""

import warnings

import numpy as np
import pandas as pd
import torch
from config import (
    CCS_CONFIG,
    DATA_CONFIG,
    MODEL_CONFIGS,
    STEERING_CONFIG,
)
from data_model_helpers import (
    create_layer_output_dir,
    create_output_dir_layers,
    extract_representations,
    load_data,
    load_model,
    save_layer_results,
    setup_layer_logging,
    setup_logging,
    setup_steering_for_layer,
    train_ccs_all_layers,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from format_results import (
    get_results_table,
    plot_all_layers_components_matrix,
    plot_pca_or_tsne_layerwise,
)

# Updated imports - separated steering modules
from steering import (  # Core steering logic only
    apply_proper_steering,
    compare_steering_layers,
)

# Changed: Import ALL analysis functions from ALL modules
from steering_analysis1_fixed import (  # Primary analysis functions
    create_best_separation_plots,
    create_comparison_results_table,
    plot_boundary_comparison_for_components,
    plot_layer_steering_effects,
    plot_steering_power,
)
from steering_analysis_all2all import (
    create_comprehensive_comparison_visualizations,
)
from steering_analysis_fixed import (  # Additional analysis functions
    plot_boundary_comparison_improved,
    plot_improved_layerwise_steering_focus,
    plot_steering_layer_analysis,
    plot_steering_power_improved,
)

# ============================================================================
# COMPREHENSIVE STEERING ANALYSIS - ALL FUNCTIONS INTEGRATED
# ============================================================================


def run_comprehensive_layer_steering_analysis(
    steering_ccs,
    X_pos,
    X_neg,
    steering_layer,
    direction_tensor,
    labels,
    train_idx,
    test_idx,
    device,
    layer_output_dir,
):
    """
    Run COMPLETE comprehensive steering analysis for a specific layer.
    """
    print(
        f"Running COMPLETE comprehensive steering analysis for layer {steering_layer}..."
    )

    # Create plots directory
    plots_dir = layer_output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Get steering alpha from config
    steering_alpha = STEERING_CONFIG.get("default_alpha", 100.0)

    # Apply proper steering with propagation
    X_pos_steered, X_neg_steered = apply_proper_steering(
        X_pos, X_neg, steering_layer, direction_tensor, steering_alpha, device
    )

    analysis_results = {}

    # =================================================================
    # 1. ENHANCED STEERING ANALYSIS (from steering_analysis1.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"ENHANCED STEERING ANALYSIS FOR LAYER {steering_layer}")
    print("=" * 60)

    # 1.1 Basic steering power plot
    print("Creating steering power analysis...")
    pos_test = torch.tensor(
        X_pos[:, steering_layer, :], dtype=torch.float32, device=device
    )
    neg_test = torch.tensor(
        X_neg[:, steering_layer, :], dtype=torch.float32, device=device
    )

    deltas = np.linspace(-3, 3, 21)
    steering_plot_path = plots_dir / f"steering_power_layer_{steering_layer}.png"

    plot_steering_power(
        ccs=steering_ccs,
        positive_statements=pos_test,
        negative_statements=neg_test,
        deltas=deltas,
        title=f"Steering Analysis - Layer {steering_layer}",
        save_path=str(steering_plot_path),
    )
    analysis_results["steering_power_plot"] = steering_plot_path

    # 1.2 Layer-wise steering effects
    print("Analyzing layer-wise steering effects...")
    layer_metrics = compare_steering_layers(X_pos, X_neg, X_pos_steered, X_neg_steered)
    layer_effects_plot = plot_layer_steering_effects(
        layer_metrics, steering_layer, plots_dir, steering_alpha, "all2all"
    )
    analysis_results["layer_effects"] = {
        "plot_path": layer_effects_plot,
        "metrics": layer_metrics,
    }

    # 1.3 Best separation component analysis
    print("Creating best separation component analysis...")
    separation_plots = create_best_separation_plots(
        positive_statements_original=X_pos[:, steering_layer, :],
        negative_statements_original=X_neg[:, steering_layer, :],
        positive_statements_steered=X_pos_steered[:, steering_layer, :],
        negative_statements_steered=X_neg_steered[:, steering_layer, :],
        y_vector=labels,
        ccs=steering_ccs,
        best_layer=steering_layer,
        steering_alpha=steering_alpha,
        plots_dir=plots_dir,
        n_components=10,
        n_plots=5,
        separation_metric="separation_index",
    )
    analysis_results["best_separation_plots"] = separation_plots

    # =================================================================
    # 2. ADDITIONAL STEERING ANALYSIS (from steering_analysis.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"ADDITIONAL STEERING ANALYSIS FOR LAYER {steering_layer}")
    print("=" * 60)

    # 2.1 Steering layer analysis plot
    print("Creating steering layer analysis plot...")
    steering_layer_analysis_path = (
        plots_dir / f"steering_layer_analysis_layer_{steering_layer}.png"
    )
    plot_steering_layer_analysis(
        layer_metrics, steering_layer, str(steering_layer_analysis_path)
    )
    analysis_results["steering_layer_analysis"] = steering_layer_analysis_path

    # 2.2 Improved layerwise steering focus
    print("Creating improved layerwise steering focus plot...")
    improved_layerwise_path = (
        plots_dir / f"improved_layerwise_steering_focus_layer_{steering_layer}.png"
    )
    plot_improved_layerwise_steering_focus(
        X_pos=X_pos,
        X_neg=X_neg,
        X_pos_steered=X_pos_steered,
        X_neg_steered=X_neg_steered,
        labels=labels,
        best_layer=steering_layer,
        steering_alpha=steering_alpha,
        save_path=str(improved_layerwise_path),
    )
    analysis_results["improved_layerwise_focus"] = improved_layerwise_path

    # 2.3 Improved steering power plot
    print("Creating improved steering power plot...")
    steering_power_improved_path = (
        plots_dir / f"steering_power_improved_layer_{steering_layer}.png"
    )
    plot_steering_power_improved(
        ccs=steering_ccs,
        X_pos=X_pos[:, steering_layer, :],
        X_neg=X_neg[:, steering_layer, :],
        direction_tensor=direction_tensor,
        best_layer=steering_layer,
        save_path=str(steering_power_improved_path),
    )
    analysis_results["steering_power_improved"] = steering_power_improved_path

    # 2.4 Improved boundary comparison
    print("Creating improved boundary comparison plot...")
    boundary_comparison_improved_path = (
        plots_dir / f"boundary_comparison_improved_layer_{steering_layer}.png"
    )
    plot_boundary_comparison_improved(
        X_pos_orig=X_pos[:, steering_layer, :],
        X_neg_orig=X_neg[:, steering_layer, :],
        X_pos_steer=X_pos_steered[:, steering_layer, :],
        X_neg_steer=X_neg_steered[:, steering_layer, :],
        labels=labels,
        ccs=steering_ccs,
        best_layer=steering_layer,
        steering_alpha=steering_alpha,
        save_path=str(boundary_comparison_improved_path),
    )
    analysis_results["boundary_comparison_improved"] = boundary_comparison_improved_path

    # =================================================================
    # 3. COMPARISON ANALYSIS (from steering_analysis1.py)
    # =================================================================
    print("\n" + "=" * 80)
    print(f"STARTING COMPARISON ANALYSIS FOR LAYER {steering_layer}")
    print("=" * 80)

    # Run the comprehensive comparison analysis
    comparison_df, orig_results, steered_results = create_comparison_results_table(
        X_pos_orig=X_pos,
        X_neg_orig=X_neg,
        X_pos_steered=X_pos_steered,
        X_neg_steered=X_neg_steered,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        best_layer=steering_layer,
        device=device,
        ccs_config=CCS_CONFIG,
        normalizing=CCS_CONFIG.get("normalizing", "mean"),
    )

    # Save main comparison CSV
    comparison_path = (
        layer_output_dir
        / f"comparison_results_layer_{steering_layer}_alpha_{steering_alpha}.csv"
    )
    comparison_df.to_csv(comparison_path)
    print(f"Comparison results saved to: {comparison_path}")

    # Save individual CSV files (as in original dump)
    print("Saving individual CSV result files...")

    # Convert results dictionaries to DataFrames using get_results_table
    from format_results import get_results_table

    orig_results_df = get_results_table(orig_results)
    steered_results_df = get_results_table(steered_results)

    # Save original results CSV
    orig_results_path = (
        layer_output_dir / f"results_original_layer_{steering_layer}.csv"
    )
    orig_results_df.to_csv(orig_results_path)
    print(f"Original results saved to: {orig_results_path}")

    # Save steered results CSV
    steered_results_path = (
        layer_output_dir
        / f"results_steered_layer_{steering_layer}_alpha_{steering_alpha}.csv"
    )
    steered_results_df.to_csv(steered_results_path)
    print(f"Steered results saved to: {steered_results_path}")

    # Save full comparison CSV (more detailed)
    full_comparison_path = (
        layer_output_dir
        / f"results_comparison_full_layer_{steering_layer}_alpha_{steering_alpha}.csv"
    )
    comparison_df.to_csv(full_comparison_path)
    print(f"Full comparison results saved to: {full_comparison_path}")

    # Calculate and save differences CSV
    print("Calculating differences between original and steered results...")
    differences_df = pd.DataFrame()

    # Calculate differences for numeric columns
    for col in orig_results_df.columns:
        if col in steered_results_df.columns and pd.api.types.is_numeric_dtype(
            orig_results_df[col]
        ):
            differences_df[f"{col}_diff"] = (
                steered_results_df[col] - orig_results_df[col]
            )
            differences_df[f"{col}_percent_change"] = (
                (
                    (steered_results_df[col] - orig_results_df[col])
                    / orig_results_df[col]
                    * 100
                )
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )

    differences_path = (
        layer_output_dir
        / f"results_differences_layer_{steering_layer}_alpha_{steering_alpha}.csv"
    )
    differences_df.to_csv(differences_path)
    print(f"Differences results saved to: {differences_path}")

    # Create comparison tables log file (as in original dump)
    print("Creating comparison tables log file...")
    logs_dir = layer_output_dir / "logs"
    comparison_log_path = (
        logs_dir
        / f"comparison_tables_layer_{steering_layer}_alpha_{steering_alpha}.log"
    )

    with open(comparison_log_path, "w") as f:
        f.write(f"Comparison Tables Log for Layer {steering_layer}\n")
        f.write("=" * 60 + "\n\n")

        f.write("ORIGINAL RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(orig_results_df.to_string())
        f.write("\n\n")

        f.write("STEERED RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(steered_results_df.to_string())
        f.write("\n\n")

        f.write("COMPARISON RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(comparison_df.to_string())
        f.write("\n\n")

        f.write("DIFFERENCES:\n")
        f.write("-" * 20 + "\n")
        f.write(differences_df.to_string())
        f.write("\n")

    print(f"Comparison tables log saved to: {comparison_log_path}")

    # =================================================================
    # 4. COMPREHENSIVE COMPARISON VISUALIZATIONS (from steering_analysis.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"COMPREHENSIVE COMPARISON VISUALIZATIONS FOR LAYER {steering_layer}")
    print("=" * 60)

    # Create comprehensive comparison visualizations
    print("Creating comprehensive comparison visualizations...")
    comprehensive_plots = create_comprehensive_comparison_visualizations(
        comparison_df=comparison_df,
        best_layer=steering_layer,
        steering_alpha=steering_alpha,
        plots_dir=plots_dir,
    )
    analysis_results["comprehensive_plots"] = comprehensive_plots

    # =================================================================
    # 5. LAYER-WISE PCA ANALYSIS (from format_results.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"LAYER-WISE ANALYSIS FOR STEERING LAYER {steering_layer}")
    print("=" * 60)

    # Original layer-wise PCA analysis
    print("Creating layer-wise PCA analysis for original model...")

    # Normalize data as required by plot_pca_or_tsne_layerwise
    X_pos_mean = X_pos.mean(0)
    X_neg_mean = X_neg.mean(0)

    X_pos_normalized = X_pos - X_pos_mean
    X_neg_normalized = X_neg - X_neg_mean

    # Check for NaN values after normalization
    if np.isnan(X_pos_normalized).any() or np.isnan(X_neg_normalized).any():
        print("Warning: Found NaN values after normalization, replacing with 0")
        X_pos_normalized = np.nan_to_num(
            X_pos_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )
        X_neg_normalized = np.nan_to_num(
            X_neg_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Create labels series for plotting
    labels_series = pd.Series(labels, name="toxicity_label")

    layerwise_original_path = plots_dir / "layerwise_pca_original.png"
    plot_pca_or_tsne_layerwise(
        X_pos=X_pos_normalized,
        X_neg=X_neg_normalized,
        hue=labels_series,
        standardize=True,
        n_components=5,
        components=[0, 1],
        mode="pca",
        plot_title=f"Layer-wise PCA Analysis - Original Model (Steering Layer {steering_layer})",
        save_path=str(layerwise_original_path),
    )

    # Normalize steered data with NaN handling
    X_pos_steered_mean = X_pos_steered.mean(0)
    X_neg_steered_mean = X_neg_steered.mean(0)

    X_pos_steered_normalized = X_pos_steered - X_pos_steered_mean
    X_neg_steered_normalized = X_neg_steered - X_neg_steered_mean

    # Check for NaN values after steered normalization
    if (
        np.isnan(X_pos_steered_normalized).any()
        or np.isnan(X_neg_steered_normalized).any()
    ):
        print("Warning: Found NaN values after steered normalization, replacing with 0")
        X_pos_steered_normalized = np.nan_to_num(
            X_pos_steered_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )
        X_neg_steered_normalized = np.nan_to_num(
            X_neg_steered_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )

    layerwise_steered_path = (
        plots_dir
        / f"layerwise_pca_steered_layer_{steering_layer}_alpha_{steering_alpha}.png"
    )
    plot_pca_or_tsne_layerwise(
        X_pos=X_pos_steered_normalized,
        X_neg=X_neg_steered_normalized,
        hue=labels_series,
        standardize=True,
        n_components=5,
        components=[0, 1],
        mode="pca",
        plot_title=f"Layer-wise PCA Analysis - Steered Model (Layer {steering_layer}, α={steering_alpha})",
        save_path=str(layerwise_steered_path),
    )

    # =================================================================
    # 6. COMPONENTS MATRIX ANALYSIS (from format_results.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"COMPONENTS MATRIX ANALYSIS FOR STEERING LAYER {steering_layer}")
    print("=" * 60)

    # Enhanced PCA components matrix analysis for ALL layers (original model)
    print(
        "Creating enhanced PCA components matrix analysis for ALL layers (original model)..."
    )

    # Create subdirectory for components matrix plots
    components_original_dir = plots_dir / "components_matrix_original"
    components_original_dir.mkdir(exist_ok=True)

    plot_all_layers_components_matrix(
        X_pos=X_pos_normalized,
        X_neg=X_neg_normalized,
        hue=labels_series,
        start_layer=0,
        standardize=True,
        n_components=5,
        mode="pca",
        plot_title_prefix=f"Original Model (Steering Layer {steering_layer}) - PCA Components Matrix",
        save_dir=components_original_dir,
    )

    # Enhanced PCA components matrix analysis for steered model (from steering layer onwards)
    print(
        f"Creating enhanced PCA components matrix analysis for steered model (from layer {steering_layer} onwards)..."
    )

    # Create subdirectory for steered components matrix plots
    components_steered_dir = (
        plots_dir / f"components_matrix_steered_from_layer_{steering_layer}"
    )
    components_steered_dir.mkdir(exist_ok=True)

    plot_all_layers_components_matrix(
        X_pos=X_pos_steered_normalized,
        X_neg=X_neg_steered_normalized,
        hue=labels_series,
        start_layer=steering_layer,
        standardize=True,
        n_components=5,
        mode="pca",
        plot_title_prefix=f"Steered Model (Layer {steering_layer}, α={steering_alpha}) - PCA Components Matrix",
        save_dir=components_steered_dir,
    )

    # =================================================================
    # 7. TRADITIONAL BOUNDARY COMPARISON (from steering_analysis1.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"TRADITIONAL BOUNDARY COMPARISON FOR LAYER {steering_layer}")
    print("=" * 60)

    if STEERING_CONFIG["plot_boundary"]:
        print("Creating traditional boundary comparison plot...")

        # Prepare steered data for comparison using simple steering
        pos_steered_simple = X_pos[:, steering_layer, :].copy()
        neg_steered_simple = X_neg[:, steering_layer, :].copy()

        # Apply simple steering (for backward compatibility)
        direction_np = direction_tensor.cpu().numpy()
        pos_steered_simple += steering_alpha * direction_np
        neg_steered_simple -= steering_alpha * direction_np

        # Create traditional comparison plot (PC0 vs PC1)
        traditional_boundary_path = (
            plots_dir
            / f"boundary_comparison_traditional_layer_{steering_layer}_alpha_{steering_alpha}.png"
        )

        # Use the traditional plot function for backward compatibility
        plot_boundary_comparison_for_components(
            positive_statements_original=X_pos[:, steering_layer, :],
            negative_statements_original=X_neg[:, steering_layer, :],
            positive_statements_steered=pos_steered_simple,
            negative_statements_steered=neg_steered_simple,
            y_vector=labels,
            ccs=steering_ccs,
            components=[0, 1],  # Traditional PC0 vs PC1
            separation_metrics={
                "silhouette_score": 0.0,
                "fisher_ratio": 0.0,
                "between_class_distance": 0.0,
                "separation_index": 0.0,
            },
            best_layer=steering_layer,
            steering_alpha=steering_alpha,
            n_components=5,
            save_path=str(traditional_boundary_path),
        )

        # Also create the basic boundary comparison plot (missing from new run)
        basic_boundary_path = (
            plots_dir
            / f"boundary_comparison_layer_{steering_layer}_alpha_{steering_alpha}.png"
        )

        # Create basic boundary comparison (simpler version)
        plot_boundary_comparison_for_components(
            positive_statements_original=X_pos[:, steering_layer, :],
            negative_statements_original=X_neg[:, steering_layer, :],
            positive_statements_steered=pos_steered_simple,
            negative_statements_steered=neg_steered_simple,
            y_vector=labels,
            ccs=steering_ccs,
            components=[0, 1],
            separation_metrics={
                "silhouette_score": 0.0,
                "fisher_ratio": 0.0,
                "between_class_distance": 0.0,
                "separation_index": 0.0,
            },
            best_layer=steering_layer,
            steering_alpha=steering_alpha,
            n_components=5,
            save_path=str(basic_boundary_path),
        )

    # =================================================================
    # 8. ADDITIONAL MISSING PLOTS
    # =================================================================
    print("\n" + "=" * 60)
    print(f"CREATING ADDITIONAL MISSING PLOTS FOR LAYER {steering_layer}")
    print("=" * 60)

    # Create steering strength analysis plot (missing from new run)
    print("Creating steering strength analysis plot...")
    steering_strength_path = (
        plots_dir / f"steering_strength_analysis_layer_{steering_layer}.png"
    )

    # This is a custom plot showing steering strength across layers
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate steering strength for each layer
    steering_strengths = []
    layer_indices = list(range(X_pos.shape[1]))

    for layer_idx in layer_indices:
        # Calculate difference between steered and original representations
        layer_diff_pos = np.mean(
            np.abs(X_pos_steered[:, layer_idx, :] - X_pos[:, layer_idx, :])
        )
        layer_diff_neg = np.mean(
            np.abs(X_neg_steered[:, layer_idx, :] - X_neg[:, layer_idx, :])
        )
        avg_diff = (layer_diff_pos + layer_diff_neg) / 2
        steering_strengths.append(avg_diff)

    # Plot steering strength
    ax.plot(
        layer_indices, steering_strengths, "b-", linewidth=2, marker="o", markersize=6
    )
    ax.axvline(
        x=steering_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Layer {steering_layer}",
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Steering Strength (Mean Absolute Difference)", fontsize=12)
    ax.set_title(
        f"Steering Strength Analysis - Layer {steering_layer} (α={steering_alpha})",
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(steering_strength_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Steering strength analysis saved to: {steering_strength_path}")
    analysis_results["steering_strength_analysis"] = steering_strength_path

    # =================================================================
    # 9. SUMMARY REPORT
    # =================================================================
    print("\n" + "=" * 60)
    print(f"COMPLETE STEERING ANALYSIS SUMMARY FOR LAYER {steering_layer}")
    print("=" * 60)

    print("✓ Enhanced steering analysis completed")
    print("✓ Additional steering analysis from steering_analysis.py completed")
    print("✓ Comprehensive comparison visualizations completed")
    print(
        "✓ Individual CSV files created (original, steered, differences, full comparison)"
    )
    print("✓ Comparison tables log file created")
    print("✓ Layer-wise PCA analysis completed")
    print("✓ Components matrix analysis completed")
    print("✓ Traditional + basic boundary comparison completed")
    print("✓ Steering strength analysis completed")

    # Count all plots created
    total_plots_created = 0

    if "best_separation_plots" in analysis_results:
        n_separation_plots = len(analysis_results["best_separation_plots"])
        print(f"✓ {n_separation_plots} best component pair separation plots created")
        total_plots_created += n_separation_plots

    if "comprehensive_plots" in analysis_results:
        n_comprehensive_plots = len(analysis_results["comprehensive_plots"])
        print(f"✓ {n_comprehensive_plots} comprehensive comparison plots created")
        total_plots_created += n_comprehensive_plots

    # Count individual analysis plots
    individual_plots = [
        "steering_power_plot",
        "steering_layer_analysis",
        "improved_layerwise_focus",
        "steering_power_improved",
        "boundary_comparison_improved",
        "steering_strength_analysis",
    ]
    n_individual_plots = sum(1 for plot in individual_plots if plot in analysis_results)
    print(f"✓ {n_individual_plots} individual analysis plots created")
    total_plots_created += n_individual_plots

    # Count additional plots (traditional + basic boundary)
    additional_plots = 2  # traditional + basic boundary
    print(f"✓ {additional_plots} additional boundary comparison plots created")
    total_plots_created += additional_plots

    print(f"\nTOTAL PLOTS CREATED FOR LAYER {steering_layer}: {total_plots_created}")

    # Count CSV files created
    csv_files = [
        "comparison_results",
        "results_original",
        "results_steered",
        "results_comparison_full",
        "results_differences",
    ]
    print(f"✓ {len(csv_files)} CSV result files created")

    # List the created files
    print(f"\nALL CREATED FILES FOR LAYER {steering_layer}:")

    print("\nCSV Files:")
    csv_file_names = [
        f"comparison_results_layer_{steering_layer}_alpha_{steering_alpha}.csv",
        f"results_original_layer_{steering_layer}.csv",
        f"results_steered_layer_{steering_layer}_alpha_{steering_alpha}.csv",
        f"results_comparison_full_layer_{steering_layer}_alpha_{steering_alpha}.csv",
        f"results_differences_layer_{steering_layer}_alpha_{steering_alpha}.csv",
    ]
    for i, csv_file in enumerate(csv_file_names):
        print(f"  {i+1}. {csv_file}")

    print("\nLog Files:")
    log_file_names = [
        f"ccs_layer_{steering_layer}_*.log",
        f"comparison_tables_layer_{steering_layer}_alpha_{steering_alpha}.log",
    ]
    for i, log_file in enumerate(log_file_names):
        print(f"  {i+1}. {log_file}")

    # List the created plots categories
    print(f"\nALL CREATED PLOTS FOR LAYER {steering_layer}:")
    plot_categories = {
        "Enhanced Separation Analysis": analysis_results.get(
            "best_separation_plots", []
        ),
        "Comprehensive Comparison": analysis_results.get("comprehensive_plots", []),
        "Individual Analysis": [
            analysis_results.get(plot)
            for plot in individual_plots
            if plot in analysis_results
        ],
    }

    for category, plots in plot_categories.items():
        if plots:
            print(f"\n{category}:")
            for i, plot_path in enumerate(plots):
                if plot_path:
                    plot_name = (
                        plot_path.name if hasattr(plot_path, "name") else str(plot_path)
                    )
                    print(f"  {i+1}. {plot_name}")

    print("\nAdditional Plots:")
    print(
        f"  1. boundary_comparison_traditional_layer_{steering_layer}_alpha_{steering_alpha}.png"
    )
    print(f"  2. boundary_comparison_layer_{steering_layer}_alpha_{steering_alpha}.png")
    print("  3. layerwise_pca_original.png")
    print(
        f"  4. layerwise_pca_steered_layer_{steering_layer}_alpha_{steering_alpha}.png"
    )

    print(f"\nAll analysis results saved to: {plots_dir}")
    print(f"All CSV files saved to: {layer_output_dir}")
    print(f"All log files saved to: {layer_output_dir / 'logs'}")
    print("=" * 60)

    return comparison_df, orig_results, steered_results


# ============================================================================
# MAIN EXECUTION FUNCTION FOR ALL-TO-ALL ANALYSIS
# ============================================================================


def run_all_layers_analysis(
    X_pos, X_neg, labels, train_idx, test_idx, device, model_config, base_output_dir
):
    """
    Run complete steering analysis for all layers.

    This is the main function that iterates through all layers and runs
    the COMPLETE pipeline for each layer as the steering layer.

    Changed: Updated to use the comprehensive analysis function
    """
    n_layers = X_pos.shape[1]

    print("\n" + "=" * 80)
    print("STARTING ALL-TO-ALL LAYER STEERING ANALYSIS WITH COMPLETE INTEGRATION")
    print(f"Total layers to analyze: {n_layers}")
    print("=" * 80)

    # Track results for all layers
    all_layer_results = {}
    all_comparison_results = {}

    # Get results table for layer selection info (but we'll use all layers)
    # This is just for maintaining compatibility with original pipeline structure
    ccs_results_global, _, _ = train_ccs_all_layers(X_pos, X_neg, labels, device)
    results_df_global = get_results_table(ccs_results_global)

    for layer_idx in range(n_layers):
        print("\n" + "=" * 100)
        print(
            f"PROCESSING LAYER {layer_idx + 1}/{n_layers} AS STEERING LAYER WITH COMPLETE ANALYSIS"
        )
        print("=" * 100)

        # Create layer-specific output directory
        layer_output_dir = create_layer_output_dir(base_output_dir, layer_idx)

        # Setup layer-specific logging
        layer_log_file, layer_logger = setup_layer_logging(layer_output_dir, layer_idx)

        layer_logger.info(
            f"Starting COMPLETE analysis for layer {layer_idx} as steering layer"
        )
        layer_logger.info(f"Layer output directory: {layer_output_dir}")

        # Setup steering for this layer
        steering_ccs, direction_tensor = setup_steering_for_layer(
            X_pos, X_neg, labels, train_idx, layer_idx, device
        )

        # Run COMPLETE comprehensive steering analysis for this layer
        print(f"Running COMPLETE analysis for layer {layer_idx}...")
        comparison_results = run_comprehensive_layer_steering_analysis(
            steering_ccs,
            X_pos,
            X_neg,
            layer_idx,  # Use layer_idx as steering layer
            direction_tensor,
            labels,
            train_idx,
            test_idx,
            device,
            layer_output_dir,
        )

        # Store comparison results
        if comparison_results is not None:
            all_comparison_results[layer_idx] = comparison_results

        # Save results for this layer
        save_layer_results(
            ccs_results_global,  # Global CCS results for all layers
            results_df_global,  # Global results table
            X_pos,
            X_neg,
            layer_idx,  # steering_layer
            direction_tensor,
            layer_output_dir,
            model_config,
            comparison_results,
        )

        # Store results summary
        all_layer_results[layer_idx] = {
            "steering_layer": layer_idx,
            "output_dir": layer_output_dir,
            "log_file": layer_log_file,
            "steering_ccs": steering_ccs,
            "direction_tensor": direction_tensor,
        }

        layer_logger.info(f"Completed COMPLETE analysis for layer {layer_idx}")

        # Print progress summary
        print(f"\n✓ Layer {layer_idx} COMPLETE analysis completed!")
        print(f"  Results saved to: {layer_output_dir}")
        print(f"  Logs saved to: {layer_log_file}")
        print(f"  Progress: {layer_idx + 1}/{n_layers} layers completed")

    return all_layer_results, all_comparison_results


def create_global_summary(base_output_dir, all_layer_results, model_config):
    """
    Create a global summary of all layer analyses.

    Parameters:
        base_output_dir: Base output directory
        all_layer_results: Dict with results for each layer
        model_config: Model configuration

    Changed: Updated summary to reflect complete analysis integration
    """
    # Create global summary file
    summary_file = base_output_dir / "all_layers_summary.txt"

    with open(summary_file, "w") as f:
        f.write(
            "CCS All-to-All Layer Steering Analysis Summary - COMPLETE INTEGRATION\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {model_config['model_name']} ({model_config['size']})\n")
        f.write(f"Total layers analyzed: {len(all_layer_results)}\n")
        f.write(f"Steering Alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}\n")
        f.write(f"Dataset: {DATA_CONFIG['dataset_name']}\n\n")

        f.write("Layer Analysis Results:\n")
        f.write("-" * 30 + "\n")

        for layer_idx, results in all_layer_results.items():
            f.write(f"\nLayer {layer_idx}:\n")
            f.write(f"  Output Directory: {results['output_dir']}\n")
            f.write(f"  Log File: {results['log_file']}\n")
            f.write("  Analysis Status: COMPLETE INTEGRATION COMPLETED\n")

            # Check if plots were created
            plots_dir = results["output_dir"] / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                f.write(f"  Total Plots Created: {len(plot_files)}\n")

                # Count different types of plots
                separation_plots = list(
                    plots_dir.glob("boundary_comparison_layer_*_PC*_rank*.png")
                )
                comparison_plots = list(plots_dir.glob("*comparison*.png"))
                analysis_plots = list(plots_dir.glob("steering_*analysis*.png"))

                f.write(f"    - Separation Plots: {len(separation_plots)}\n")
                f.write(f"    - Comparison Plots: {len(comparison_plots)}\n")
                f.write(f"    - Analysis Plots: {len(analysis_plots)}\n")

            # Check if comparison results exist
            results_dir = results["output_dir"] / "results"
            comparison_file = results_dir / "comparison_results.pkl"
            if comparison_file.exists():
                f.write("  Comprehensive Analysis: Yes\n")
            else:
                f.write("  Comprehensive Analysis: No\n")

        f.write("\n\nDirectory Structure:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{base_output_dir.name}/\n")
        for layer_idx in sorted(all_layer_results.keys()):
            f.write(f"├── layer_{layer_idx}/\n")
            f.write("│   ├── plots/ (COMPLETE set of analysis plots)\n")
            f.write(
                "│   │   ├── boundary_comparison_layer_X_alpha_Y_PCi_PCj_rankN.png (5 files)\n"
            )
            f.write("│   │   ├── steering_*_analysis_*.png (multiple files)\n")
            f.write("│   │   ├── *comparison*.png (multiple files)\n")
            f.write("│   │   ├── layerwise_pca_*.png\n")
            f.write("│   │   └── components_matrix_*/ (subdirectories)\n")
            f.write("│   ├── logs/\n")
            f.write("│   └── results/\n")
        f.write("└── all_layers_summary.txt\n")

    print(f"Global summary saved to: {summary_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function for all-to-all layer analysis with COMPLETE integration."""
    print("=" * 80)
    print("CCS All-to-All Layer Steering Pipeline with COMPLETE ANALYSIS INTEGRATION")
    print("=" * 80)

    # Get active model config
    active_models = [k for k, v in MODEL_CONFIGS.items()]
    if not active_models:
        raise ValueError(
            "No active model configuration found. Please uncomment one model in MODEL_CONFIGS."
        )

    model_key = active_models[0]
    model_config = MODEL_CONFIGS[model_key]

    # Create output directory with model name and size
    output_dir = create_output_dir_layers(model_config)

    # Setup main logging (this will be the overall pipeline log)
    log_file, stdout_log = setup_logging(output_dir)

    print(
        f"Starting COMPLETE all-to-all pipeline for model: {model_config['model_name']} ({model_config['size']})"
    )
    print(f"Base output directory: {output_dir}")

    # Print configuration summary
    print("\n" + "=" * 80)
    print("ALL-TO-ALL PIPELINE CONFIGURATION SUMMARY - COMPLETE INTEGRATION")
    print("=" * 80)
    print(f"Data: {DATA_CONFIG['dataset_name']}")
    print(f"Model: {model_config['model_name']} ({model_config['size']})")
    print(f"CCS Lambda: {CCS_CONFIG['lambda_classification']}")
    print(f"Normalizing: {CCS_CONFIG['normalizing']}")
    print(f"Steering Alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}")

    # Load data
    positive_texts, negative_texts, labels = load_data()

    # Load model
    model, tokenizer, device, model_config = load_model()

    # Extract representations
    X_pos, X_neg = extract_representations(
        model, tokenizer, positive_texts, negative_texts, device, model_config
    )

    # Get train/test split (same split used for all layers)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=labels,
    )

    print("\nDataset split:")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Testing samples: {len(test_idx)}")
    print(f"  Total layers to analyze: {X_pos.shape[1]}")

    # Run all-to-all layer analysis with COMPLETE integration
    all_layer_results, all_comparison_results = run_all_layers_analysis(
        X_pos, X_neg, labels, train_idx, test_idx, device, model_config, output_dir
    )

    # Create global summary
    create_global_summary(output_dir, all_layer_results, model_config)

    # Print comprehensive completion summary
    print("\n" + "=" * 80)
    print("ALL-TO-ALL PIPELINE WITH COMPLETE INTEGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Model: {model_config['model_name']} ({model_config['size']})")
    print(f"Layers analyzed: {len(all_layer_results)}")
    print(f"Steering alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}")
    print(f"Base results directory: {output_dir}")
    print(f"Main logs saved to: {output_dir / 'logs'}")

    # Enhanced summary for COMPLETE integration
    print("\nCOMPLETE INTEGRATION ANALYSIS OUTPUTS:")
    print("  📊 Enhanced separation analysis plots (5 per layer)")
    print("  📈 Comprehensive comparison visualizations (multiple per layer)")
    print("  📉 Layer-wise PCA comparisons (original + steered per layer)")
    print("  📊 Components matrix analysis (original + steered per layer)")
    print("  🎯 Traditional + improved boundary comparisons per layer")
    print("  📈 Steering power analysis (basic + improved per layer)")
    print("  📊 Layer steering effects analysis per layer")
    print("  🔍 Improved layerwise steering focus per layer")

    # Count total analysis outputs
    total_comparison_layers = len(all_comparison_results)
    print(f"\nTOTAL LAYERS WITH COMPLETE ANALYSIS: {total_comparison_layers}")

    if all_comparison_results:
        print("\nCOMPLETE ANALYSIS OUTPUTS PER LAYER:")
        print("  📊 CSV comparison tables")
        print("  📝 Pretty-printed log tables")
        print("  📈 Heatmap comparisons")
        print("  📉 Line plot comparisons")
        print("  📊 Bar plot analyses")
        print("  🔍 Difference analysis plots")
        print("  🎯 Component separation rankings")
        print("  📈 Steering power visualizations")
        print("  📊 Layer effects quantification")

    print("\nDIRECTORY STRUCTURE:")
    print(f"  {output_dir.name}/")
    for layer_idx in sorted(all_layer_results.keys()):
        print(f"  ├── layer_{layer_idx}/")
        print("  │   ├── plots/ (COMPLETE analysis plots)")
        print(
            "  │   │   ├── boundary_comparison_layer_X_alpha_Y_PCi_PCj_rankN.png (5 files)"
        )
        print("  │   │   ├── steering_power_*.png (2 versions)")
        print("  │   │   ├── steering_layer_analysis_*.png")
        print("  │   │   ├── improved_layerwise_steering_focus_*.png")
        print("  │   │   ├── boundary_comparison_improved_*.png")
        print("  │   │   ├── *comparison*.png (comprehensive set)")
        print("  │   │   ├── layerwise_pca_*.png (original + steered)")
        print("  │   │   └── components_matrix_*/ (original + steered)")
        print("  │   ├── logs/")
        print("  │   └── results/")
    print("  └── all_layers_summary.txt")

    # Show comprehensive findings if analysis was run
    if all_comparison_results:
        print("\nCOMPREHENSIVE ANALYSIS FINDINGS (showing first 3 layers):")

        for layer_idx in sorted(list(all_comparison_results.keys())[:3]):
            comparison_results = all_comparison_results[layer_idx]
            if comparison_results is not None:
                comparison_df, _, _ = comparison_results

                print(f"\n  Layer {layer_idx} COMPLETE Analysis Results:")

                # Get base metric names
                base_metrics = []
                for col in comparison_df.columns:
                    if col.endswith("_original"):
                        base_metrics.append(col.replace("_original", ""))

                # Show top 3 metrics for this layer
                for metric in base_metrics[:3]:
                    orig_val = comparison_df.loc[layer_idx, f"{metric}_original"]
                    steered_val = comparison_df.loc[layer_idx, f"{metric}_steered"]
                    pct_change = comparison_df.loc[
                        layer_idx, f"{metric}_percent_change"
                    ]

                    print(
                        f"    {metric}: {orig_val:.4f} → {steered_val:.4f} ({pct_change:+.1f}%)"
                    )

    print("\n📊 NEW ENHANCED FEATURES:")
    print("   🔍 5 Best component pair separation plots per layer")
    print("   📈 Improved steering power analysis")
    print("   📊 Comprehensive comparison visualizations")
    print("   🎯 Enhanced boundary comparison analysis")
    print("   📉 Layer-wise steering effects quantification")

    print("\n📊 ALL-TO-ALL COMPLETE ANALYSIS FINISHED!")
    print(
        "   Every layer has been analyzed as a steering layer with COMPLETE functionality!"
    )
    print("   All analysis modules have been fully integrated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
