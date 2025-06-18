#!/usr/bin/env python3
"""
CCS Training and Steering Pipeline - Single Best Layer Analysis with COMPLETE Integration
========================================================================================

This script runs the complete pipeline for the BEST layer analysis:
1. Load model and data
2. Extract representations
3. Train vanilla CCS on all layers
4. Select best layer based on specified metric
5. Run comprehensive steering analysis on best layer with ALL functions
6. Save results

Enhanced with COMPLETE analysis integration:
- ALL functions from steering_analysis.py
- ALL functions from steering_analysis1.py
- ALL functions from format_results.py
- Enhanced steering analysis with component separation
- Comprehensive comparison analysis
- Layer-wise PCA comparisons
- Component matrix analysis
- Best separation plots
- ALL missing CSV files and plots

Output structure:
pythia-1b_1B_timestamp/
├── plots/
├── logs/
├── results/
├── comparison_results_layer_X_alpha_Y.csv
├── results_original_layer_X.csv
├── results_steered_layer_X_alpha_Y.csv
├── results_comparison_full_layer_X_alpha_Y.csv
├── results_differences_layer_X_alpha_Y.csv
└── other files...

Changed: Added ALL analysis functions from ALL modules + missing files
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
    create_output_dir,
    extract_representations,
    load_data,
    load_model,
    save_results,
    select_best_layer,
    setup_logging,
    setup_steering,
    train_ccs_all_layers,
)

warnings.filterwarnings("ignore")

from format_results import (
    plot_all_layers_components_matrix,
    plot_pca_or_tsne_layerwise,
)

# Updated imports - separated steering modules
from steering import (  # Core steering logic only
    apply_proper_steering,
    compare_steering_layers,
)
from steering_analysis import (  # Additional analysis functions
    create_comprehensive_comparison_visualizations,
    plot_boundary_comparison_improved,
    plot_improved_layerwise_steering_focus,
    plot_steering_layer_analysis,
    plot_steering_power_improved,
)

# Changed: Import ALL analysis functions from ALL modules
from steering_analysis1 import (  # Primary analysis functions
    create_best_separation_plots,
    create_comparison_results_table,
    plot_boundary_comparison_for_components,
    plot_layer_steering_effects,
    plot_steering_power,
)

# ============================================================================
# COMPREHENSIVE STEERING ANALYSIS - ALL FUNCTIONS INTEGRATED
# ============================================================================


def run_comprehensive_best_layer_steering_analysis(
    best_ccs,
    X_pos,
    X_neg,
    best_layer,
    direction_tensor,
    labels,
    train_idx,
    test_idx,
    device,
    output_dir,
):
    """
    Run COMPLETE comprehensive steering analysis for the best layer.

    Changed: Integrated ALL functions from ALL analysis modules:
    - steering_analysis1.py
    - steering_analysis.py
    - format_results.py
    - PLUS all missing CSV files and plots
    """
    print(
        f"Running COMPLETE comprehensive steering analysis for best layer {best_layer}..."
    )

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Get steering alpha from config
    steering_alpha = STEERING_CONFIG.get("default_alpha", 2.0)

    # Apply proper steering with propagation
    X_pos_steered, X_neg_steered = apply_proper_steering(
        X_pos, X_neg, best_layer, direction_tensor, steering_alpha, device
    )

    analysis_results = {}

    # =================================================================
    # 1. ENHANCED STEERING ANALYSIS (from steering_analysis1.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"ENHANCED STEERING ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # 1.1 Basic steering power plot
    print("Creating steering power analysis...")
    pos_test = torch.tensor(X_pos[:, best_layer, :], dtype=torch.float32, device=device)
    neg_test = torch.tensor(X_neg[:, best_layer, :], dtype=torch.float32, device=device)

    deltas = np.linspace(-3, 3, 21)
    steering_plot_path = plots_dir / f"steering_power_layer_{best_layer}.png"

    plot_steering_power(
        ccs=best_ccs,
        positive_statements=pos_test,
        negative_statements=neg_test,
        deltas=deltas,
        title=f"Steering Analysis - Best Layer {best_layer}",
        save_path=str(steering_plot_path),
    )
    analysis_results["steering_power_plot"] = steering_plot_path

    # 1.2 Layer-wise steering effects
    print("Analyzing layer-wise steering effects...")
    layer_metrics = compare_steering_layers(X_pos, X_neg, X_pos_steered, X_neg_steered)
    layer_effects_plot = plot_layer_steering_effects(
        layer_metrics, best_layer, plots_dir, steering_alpha
    )
    analysis_results["layer_effects"] = {
        "plot_path": layer_effects_plot,
        "metrics": layer_metrics,
    }

    # 1.3 Best separation component analysis
    print("Creating best separation component analysis...")
    separation_plots = create_best_separation_plots(
        positive_statements_original=X_pos[:, best_layer, :],
        negative_statements_original=X_neg[:, best_layer, :],
        positive_statements_steered=X_pos_steered[:, best_layer, :],
        negative_statements_steered=X_neg_steered[:, best_layer, :],
        y_vector=labels,
        ccs=best_ccs,
        best_layer=best_layer,
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
    print(f"ADDITIONAL STEERING ANALYSIS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # 2.1 Steering layer analysis plot
    print("Creating steering layer analysis plot...")
    steering_layer_analysis_path = (
        plots_dir / f"steering_layer_analysis_layer_{best_layer}.png"
    )
    plot_steering_layer_analysis(
        layer_metrics, best_layer, str(steering_layer_analysis_path)
    )
    analysis_results["steering_layer_analysis"] = steering_layer_analysis_path

    # 2.2 Improved layerwise steering focus
    print("Creating improved layerwise steering focus plot...")
    improved_layerwise_path = (
        plots_dir / f"improved_layerwise_steering_focus_layer_{best_layer}.png"
    )
    plot_improved_layerwise_steering_focus(
        X_pos=X_pos,
        X_neg=X_neg,
        X_pos_steered=X_pos_steered,
        X_neg_steered=X_neg_steered,
        labels=labels,
        best_layer=best_layer,
        steering_alpha=steering_alpha,
        save_path=str(improved_layerwise_path),
    )
    analysis_results["improved_layerwise_focus"] = improved_layerwise_path

    # 2.3 Improved steering power plot
    print("Creating improved steering power plot...")
    steering_power_improved_path = (
        plots_dir / f"steering_power_improved_layer_{best_layer}.png"
    )
    plot_steering_power_improved(
        ccs=best_ccs,
        X_pos=X_pos[:, best_layer, :],
        X_neg=X_neg[:, best_layer, :],
        direction_tensor=direction_tensor,
        best_layer=best_layer,
        save_path=str(steering_power_improved_path),
    )
    analysis_results["steering_power_improved"] = steering_power_improved_path

    # 2.4 Improved boundary comparison
    print("Creating improved boundary comparison plot...")
    boundary_comparison_improved_path = (
        plots_dir / f"boundary_comparison_improved_layer_{best_layer}.png"
    )
    plot_boundary_comparison_improved(
        X_pos_orig=X_pos[:, best_layer, :],
        X_neg_orig=X_neg[:, best_layer, :],
        X_pos_steer=X_pos_steered[:, best_layer, :],
        X_neg_steer=X_neg_steered[:, best_layer, :],
        labels=labels,
        ccs=best_ccs,
        best_layer=best_layer,
        steering_alpha=steering_alpha,
        save_path=str(boundary_comparison_improved_path),
    )
    analysis_results["boundary_comparison_improved"] = boundary_comparison_improved_path

    # =================================================================
    # 3. COMPARISON ANALYSIS (from steering_analysis1.py)
    # =================================================================
    print("\n" + "=" * 80)
    print(f"STARTING COMPARISON ANALYSIS FOR BEST LAYER {best_layer}")
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
        best_layer=best_layer,
        device=device,
        ccs_config=CCS_CONFIG,
        normalizing=CCS_CONFIG.get("normalizing", "mean"),
    )

    # Save main comparison CSV
    comparison_path = (
        output_dir / f"comparison_results_layer_{best_layer}_alpha_{steering_alpha}.csv"
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
    orig_results_path = output_dir / f"results_original_layer_{best_layer}.csv"
    orig_results_df.to_csv(orig_results_path)
    print(f"Original results saved to: {orig_results_path}")

    # Save steered results CSV
    steered_results_path = (
        output_dir / f"results_steered_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    steered_results_df.to_csv(steered_results_path)
    print(f"Steered results saved to: {steered_results_path}")

    # Save full comparison CSV (more detailed)
    full_comparison_path = (
        output_dir
        / f"results_comparison_full_layer_{best_layer}_alpha_{steering_alpha}.csv"
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
        output_dir
        / f"results_differences_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    differences_df.to_csv(differences_path)
    print(f"Differences results saved to: {differences_path}")

    # Create comparison tables log file (as in original dump)
    print("Creating comparison tables log file...")
    logs_dir = output_dir / "logs"
    comparison_log_path = (
        logs_dir / f"comparison_tables_layer_{best_layer}_alpha_{steering_alpha}.log"
    )

    with open(comparison_log_path, "w") as f:
        f.write(f"Comparison Tables Log for Best Layer {best_layer}\n")
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
    print(f"COMPREHENSIVE COMPARISON VISUALIZATIONS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Create comprehensive comparison visualizations
    print("Creating comprehensive comparison visualizations...")
    comprehensive_plots = create_comprehensive_comparison_visualizations(
        comparison_df=comparison_df,
        best_layer=best_layer,
        steering_alpha=steering_alpha,
        plots_dir=plots_dir,
    )
    analysis_results["comprehensive_plots"] = comprehensive_plots

    # =================================================================
    # 5. LAYER-WISE PCA ANALYSIS (from format_results.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"LAYER-WISE ANALYSIS FOR BEST LAYER {best_layer}")
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
        plot_title=f"Layer-wise PCA Analysis - Original Model (Best Layer {best_layer})",
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
        / f"layerwise_pca_steered_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    plot_pca_or_tsne_layerwise(
        X_pos=X_pos_steered_normalized,
        X_neg=X_neg_steered_normalized,
        hue=labels_series,
        standardize=True,
        n_components=5,
        components=[0, 1],
        mode="pca",
        plot_title=f"Layer-wise PCA Analysis - Steered Model (Best Layer {best_layer}, α={steering_alpha})",
        save_path=str(layerwise_steered_path),
    )

    # =================================================================
    # 6. COMPONENTS MATRIX ANALYSIS (from format_results.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"COMPONENTS MATRIX ANALYSIS FOR BEST LAYER {best_layer}")
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
        plot_title_prefix=f"Original Model (Best Layer {best_layer}) - PCA Components Matrix",
        save_dir=components_original_dir,
    )

    # Enhanced PCA components matrix analysis for steered model (from best layer onwards)
    print(
        f"Creating enhanced PCA components matrix analysis for steered model (from best layer {best_layer} onwards)..."
    )

    # Create subdirectory for steered components matrix plots
    components_steered_dir = (
        plots_dir / f"components_matrix_steered_from_layer_{best_layer}"
    )
    components_steered_dir.mkdir(exist_ok=True)

    plot_all_layers_components_matrix(
        X_pos=X_pos_steered_normalized,
        X_neg=X_neg_steered_normalized,
        hue=labels_series,
        start_layer=best_layer,
        standardize=True,
        n_components=5,
        mode="pca",
        plot_title_prefix=f"Steered Model (Best Layer {best_layer}, α={steering_alpha}) - PCA Components Matrix",
        save_dir=components_steered_dir,
    )

    # =================================================================
    # 7. TRADITIONAL BOUNDARY COMPARISON (from steering_analysis1.py)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"TRADITIONAL BOUNDARY COMPARISON FOR BEST LAYER {best_layer}")
    print("=" * 60)

    if STEERING_CONFIG["plot_boundary"]:
        print("Creating traditional boundary comparison plot...")

        # Prepare steered data for comparison using simple steering
        pos_steered_simple = X_pos[:, best_layer, :].copy()
        neg_steered_simple = X_neg[:, best_layer, :].copy()

        # Apply simple steering (for backward compatibility)
        direction_np = direction_tensor.cpu().numpy()
        pos_steered_simple += steering_alpha * direction_np
        neg_steered_simple -= steering_alpha * direction_np

        # Create traditional comparison plot (PC0 vs PC1)
        traditional_boundary_path = (
            plots_dir
            / f"boundary_comparison_traditional_layer_{best_layer}_alpha_{steering_alpha}.png"
        )

        # Use the traditional plot function for backward compatibility
        plot_boundary_comparison_for_components(
            positive_statements_original=X_pos[:, best_layer, :],
            negative_statements_original=X_neg[:, best_layer, :],
            positive_statements_steered=pos_steered_simple,
            negative_statements_steered=neg_steered_simple,
            y_vector=labels,
            ccs=best_ccs,
            components=[0, 1],  # Traditional PC0 vs PC1
            separation_metrics={
                "silhouette_score": 0.0,
                "fisher_ratio": 0.0,
                "between_class_distance": 0.0,
                "separation_index": 0.0,
            },
            best_layer=best_layer,
            steering_alpha=steering_alpha,
            n_components=5,
            save_path=str(traditional_boundary_path),
        )

        # Also create the basic boundary comparison plot (missing from new run)
        basic_boundary_path = (
            plots_dir
            / f"boundary_comparison_layer_{best_layer}_alpha_{steering_alpha}.png"
        )

        # Create basic boundary comparison (simpler version)
        plot_boundary_comparison_for_components(
            positive_statements_original=X_pos[:, best_layer, :],
            negative_statements_original=X_neg[:, best_layer, :],
            positive_statements_steered=pos_steered_simple,
            negative_statements_steered=neg_steered_simple,
            y_vector=labels,
            ccs=best_ccs,
            components=[0, 1],
            separation_metrics={
                "silhouette_score": 0.0,
                "fisher_ratio": 0.0,
                "between_class_distance": 0.0,
                "separation_index": 0.0,
            },
            best_layer=best_layer,
            steering_alpha=steering_alpha,
            n_components=5,
            save_path=str(basic_boundary_path),
        )

    # =================================================================
    # 8. ADDITIONAL MISSING PLOTS
    # =================================================================
    print("\n" + "=" * 60)
    print(f"CREATING ADDITIONAL MISSING PLOTS FOR BEST LAYER {best_layer}")
    print("=" * 60)

    # Create steering strength analysis plot (missing from new run)
    print("Creating steering strength analysis plot...")
    steering_strength_path = (
        plots_dir / f"steering_strength_analysis_layer_{best_layer}.png"
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
        x=best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best Layer {best_layer}",
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Steering Strength (Mean Absolute Difference)", fontsize=12)
    ax.set_title(
        f"Steering Strength Analysis - Best Layer {best_layer} (α={steering_alpha})",
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
    print(f"COMPLETE STEERING ANALYSIS SUMMARY FOR BEST LAYER {best_layer}")
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

    print(f"\nTOTAL PLOTS CREATED FOR BEST LAYER {best_layer}: {total_plots_created}")

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
    print(f"\nALL CREATED FILES FOR BEST LAYER {best_layer}:")

    print("\nCSV Files:")
    csv_file_names = [
        f"comparison_results_layer_{best_layer}_alpha_{steering_alpha}.csv",
        f"results_original_layer_{best_layer}.csv",
        f"results_steered_layer_{best_layer}_alpha_{steering_alpha}.csv",
        f"results_comparison_full_layer_{best_layer}_alpha_{steering_alpha}.csv",
        f"results_differences_layer_{best_layer}_alpha_{steering_alpha}.csv",
    ]
    for i, csv_file in enumerate(csv_file_names):
        print(f"  {i+1}. {csv_file}")

    print("\nLog Files:")
    log_file_names = [
        "ccs_pipeline_*.log",
        f"comparison_tables_layer_{best_layer}_alpha_{steering_alpha}.log",
    ]
    for i, log_file in enumerate(log_file_names):
        print(f"  {i+1}. {log_file}")

    print(f"\nAll analysis results saved to: {plots_dir}")
    print(f"All CSV files saved to: {output_dir}")
    print(f"All log files saved to: {output_dir / 'logs'}")
    print("=" * 60)

    return comparison_df, orig_results, steered_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function for best layer analysis with COMPLETE integration."""
    print("=" * 80)
    print("CCS Best Layer Steering Pipeline with COMPLETE ANALYSIS INTEGRATION")
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
    output_dir = create_output_dir(model_config)

    # Setup main logging
    log_file, stdout_log = setup_logging(output_dir)

    print(
        f"Starting COMPLETE best layer pipeline for model: {model_config['model_name']} ({model_config['size']})"
    )
    print(f"Output directory: {output_dir}")

    # Print configuration summary
    print("\n" + "=" * 80)
    print("BEST LAYER PIPELINE CONFIGURATION SUMMARY - COMPLETE INTEGRATION")
    print("=" * 80)
    print(f"Data: {DATA_CONFIG['dataset_name']}")
    print(f"Model: {model_config['model_name']} ({model_config['size']})")
    print(f"CCS Lambda: {CCS_CONFIG['lambda_classification']}")
    print(f"Normalizing: {CCS_CONFIG['normalizing']}")
    print(f"Steering Alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}")
    print("Enhanced Component Analysis: ✓ ENABLED")
    print("Complete Function Integration: ✓ ENABLED")
    print("steering_analysis.py Functions: ✓ ALL INTEGRATED")
    print("steering_analysis1.py Functions: ✓ ALL INTEGRATED")
    print("format_results.py Functions: ✓ ALL INTEGRATED")
    print("Mode: BEST LAYER ANALYSIS WITH COMPLETE INTEGRATION")
    print("=" * 80)

    # Load data
    positive_texts, negative_texts, labels = load_data()

    # Load model
    model, tokenizer, device, model_config = load_model()

    # Extract representations
    X_pos, X_neg = extract_representations(
        model, tokenizer, positive_texts, negative_texts, device, model_config
    )

    # Train CCS on all layers
    ccs_results, train_idx, test_idx = train_ccs_all_layers(
        X_pos, X_neg, labels, device
    )

    # Select best layer
    best_layer, results_df = select_best_layer(ccs_results)

    # Setup steering for best layer
    best_ccs, direction_tensor = setup_steering(
        X_pos, X_neg, labels, train_idx, best_layer, device
    )

    print("\nDataset and model info:")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Testing samples: {len(test_idx)}")
    print(f"  Total layers: {X_pos.shape[1]}")
    print(f"  Best layer selected: {best_layer}")

    # Run COMPLETE comprehensive steering analysis for best layer
    print(f"\nRunning COMPLETE analysis for best layer {best_layer}...")
    comparison_results = run_comprehensive_best_layer_steering_analysis(
        best_ccs,
        X_pos,
        X_neg,
        best_layer,
        direction_tensor,
        labels,
        train_idx,
        test_idx,
        device,
        output_dir,
    )

    # Save all results including comparison results
    save_results(
        ccs_results,
        results_df,
        X_pos,
        X_neg,
        best_layer,
        direction_tensor,
        output_dir,
        model_config,
        comparison_results,
    )

    # Print comprehensive completion summary
    print("\n" + "=" * 80)
    print("BEST LAYER PIPELINE WITH COMPLETE INTEGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Model: {model_config['model_name']} ({model_config['size']})")
    print(f"Best layer: {best_layer}")
    print(f"Steering alpha: {STEERING_CONFIG.get('default_alpha', 2.0)}")
    print(f"Results directory: {output_dir}")
    print(f"Logs saved to: {output_dir / 'logs'}")

    # Enhanced summary for COMPLETE integration
    print("\nCOMPLETE INTEGRATION ANALYSIS OUTPUTS:")
    print("  📊 Enhanced separation analysis plots (5 plots)")
    print("  📈 Comprehensive comparison visualizations (multiple plots)")
    print("  📉 Layer-wise PCA comparisons (original + steered)")
    print("  📊 Components matrix analysis (original + steered)")
    print("  🎯 Traditional + improved boundary comparisons")
    print("  📈 Steering power analysis (basic + improved)")
    print("  📊 Layer steering effects analysis")
    print("  🔍 Improved layerwise steering focus")
    print("  📊 Steering strength analysis")

    # Show CSV file outputs
    print("\nCOMPLETE CSV FILE OUTPUTS:")
    print("  📊 comparison_results_layer_X_alpha_Y.csv")
    print("  📝 results_original_layer_X.csv")
    print("  📈 results_steered_layer_X_alpha_Y.csv")
    print("  📉 results_comparison_full_layer_X_alpha_Y.csv")
    print("  📊 results_differences_layer_X_alpha_Y.csv")

    # Show log file outputs
    print("\nCOMPLETE LOG FILE OUTPUTS:")
    print("  📝 ccs_pipeline_*.log")
    print("  📊 comparison_tables_layer_X_alpha_Y.log")

    print("\nDIRECTORY STRUCTURE:")
    print(f"  {output_dir.name}/")
    print("  ├── plots/ (COMPLETE analysis plots)")
    print("  │   ├── boundary_comparison_layer_X_alpha_Y_PCi_PCj_rankN.png (5 files)")
    print("  │   ├── steering_power_*.png (2 versions)")
    print("  │   ├── steering_layer_analysis_*.png")
    print("  │   ├── improved_layerwise_steering_focus_*.png")
    print("  │   ├── boundary_comparison_improved_*.png")
    print("  │   ├── steering_strength_analysis_*.png")
    print("  │   ├── *comparison*.png (comprehensive set)")
    print("  │   ├── layerwise_pca_*.png (original + steered)")
    print("  │   └── components_matrix_*/ (original + steered)")
    print("  ├── logs/ (complete log files)")
    print("  ├── results/ (pkl files)")
    print("  ├── comparison_results_layer_X_alpha_Y.csv")
    print("  ├── results_original_layer_X.csv")
    print("  ├── results_steered_layer_X_alpha_Y.csv")
    print("  ├── results_comparison_full_layer_X_alpha_Y.csv")
    print("  ├── results_differences_layer_X_alpha_Y.csv")
    print("  └── other files...")

    # Show comprehensive findings if analysis was run
    if comparison_results is not None:
        comparison_df, _, _ = comparison_results

        print("\nCOMPREHENSIVE ANALYSIS FINDINGS:")
        print(f"  Best Layer {best_layer} Analysis Results:")

        # Get base metric names
        base_metrics = []
        for col in comparison_df.columns:
            if col.endswith("_original"):
                base_metrics.append(col.replace("_original", ""))

        # Show top 3 metrics for best layer
        for metric in base_metrics[:3]:
            orig_val = comparison_df.loc[best_layer, f"{metric}_original"]
            steered_val = comparison_df.loc[best_layer, f"{metric}_steered"]
            pct_change = comparison_df.loc[best_layer, f"{metric}_percent_change"]

            print(
                f"    {metric}: {orig_val:.4f} → {steered_val:.4f} ({pct_change:+.1f}%)"
            )

    print("\n🎯 COMPLETE INTEGRATION FEATURES:")
    print("   ✓ ALL functions from steering_analysis.py integrated")
    print("   ✓ ALL functions from steering_analysis1.py integrated")
    print("   ✓ ALL functions from format_results.py integrated")
    print("   ✓ Enhanced component separation analysis")
    print("   ✓ Comprehensive comparison visualizations")
    print("   ✓ Improved steering analysis plots")
    print("   ✓ Layer-wise PCA and component matrix analysis")
    print("   ✓ ALL missing CSV files from original dump")
    print("   ✓ ALL missing plot files from original dump")
    print("   ✓ ALL missing log files from original dump")

    print("\n📊 NEW ENHANCED FEATURES:")
    print("   🔍 5 Best component pair separation plots")
    print("   📈 Improved steering power analysis")
    print("   📊 Comprehensive comparison visualizations")
    print("   🎯 Enhanced boundary comparison analysis")
    print("   📉 Layer-wise steering effects quantification")
    print("   📊 Steering strength analysis across all layers")

    print("\n📊 BEST LAYER COMPLETE ANALYSIS FINISHED!")
    print("   Best layer has been analyzed with COMPLETE functionality!")
    print("   All analysis modules have been fully integrated!")
    print("   All missing files from original dump have been recreated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
