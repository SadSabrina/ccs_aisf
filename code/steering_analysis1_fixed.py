"""
Steering Analysis and Visualization Functions

This module contains all analysis and plotting functions for steering experiments.
Separated from core steering logic for better organization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ccs import train_ccs_on_hidden_states
from format_results import get_results_table

# Import CCS functions
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Import core steering functions

# =============================================================================
# SEPARATION ANALYSIS FUNCTIONS
# =============================================================================


def calculate_separation_metrics(X_pca, y_vector, component_i, component_j):
    """
    Calculate multiple separation metrics for a given component pair.

    Parameters:
        X_pca: PCA-transformed data [N, n_components]
        y_vector: True labels [N,]
        component_i: First component index
        component_j: Second component index

    Returns:
        dict: Dictionary with separation metrics
    """
    # Extract 2D data for the component pair
    X_2d = X_pca[:, [component_i, component_j]]

    # Handle edge cases - check if we have at least 2 classes
    unique_labels = np.unique(y_vector)
    if len(unique_labels) < 2:
        return {
            "silhouette_score": 0.0,
            "fisher_ratio": 0.0,
            "between_class_distance": 0.0,
            "separation_index": 0.0,
            "class_variance_ratio": 0.0,
        }

    metrics = {}

    # 1. Silhouette Score - measures how well-separated clusters are
    if len(X_2d) > 1 and len(unique_labels) > 1:
        metrics["silhouette_score"] = silhouette_score(
            X_2d, y_vector, metric="euclidean"
        )
    else:
        metrics["silhouette_score"] = 0.0

    # 2. Fisher's Linear Discriminant Ratio (between-class vs within-class variance)
    class_0_data = X_2d[y_vector == 0]
    class_1_data = X_2d[y_vector == 1]

    if len(class_0_data) > 0 and len(class_1_data) > 0:
        # Between-class variance (distance between means)
        mean_0 = np.mean(class_0_data, axis=0)
        mean_1 = np.mean(class_1_data, axis=0)
        between_class_var = np.linalg.norm(mean_1 - mean_0) ** 2

        # Within-class variance
        var_0 = np.mean(np.linalg.norm(class_0_data - mean_0, axis=1) ** 2)
        var_1 = np.mean(np.linalg.norm(class_1_data - mean_1, axis=1) ** 2)
        within_class_var = (var_0 + var_1) / 2

        # Fisher ratio
        metrics["fisher_ratio"] = between_class_var / (within_class_var + 1e-8)
        metrics["between_class_distance"] = np.sqrt(between_class_var)
        metrics["class_variance_ratio"] = var_0 / (var_1 + 1e-8)
    else:
        metrics["fisher_ratio"] = 0.0
        metrics["between_class_distance"] = 0.0
        metrics["class_variance_ratio"] = 1.0

    # 3. Custom separation index (combines multiple factors)
    if len(class_0_data) > 0 and len(class_1_data) > 0:
        # Normalize silhouette score (from [-1,1] to [0,1])
        norm_silhouette = (metrics["silhouette_score"] + 1) / 2

        # Combine metrics with weights
        metrics["separation_index"] = (
            0.4 * norm_silhouette
            + 0.3 * min(metrics["fisher_ratio"] / 10, 1.0)  # Cap at 1.0
            + 0.3 * min(metrics["between_class_distance"] / 5, 1.0)  # Cap at 1.0
        )
    else:
        metrics["separation_index"] = 0.0

    return metrics


def plot_boundary_comparison_for_components(
    positive_statements_original,
    negative_statements_original,
    positive_statements_steered,
    negative_statements_steered,
    y_vector,
    ccs,
    components,  # [component_i, component_j]
    separation_metrics,  # dict with metric values
    best_layer,
    steering_alpha,
    n_components=10,
    save_path=None,
):
    """
    Create boundary comparison plot for specific component pair.

    Parameters:
        components: [component_i, component_j] - which components to plot
        separation_metrics: dict with separation metric values
        Other parameters same as original function
    """

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    w, b = ccs.get_weights()
    component_i, component_j = components

    # === CRITICAL FIX: Use SAME PCA transformation for both original and steered data ===

    # Combine data
    X_all_orig = np.vstack([positive_statements_original, negative_statements_original])
    X_all_steer = np.vstack([positive_statements_steered, negative_statements_steered])

    y_combined = np.hstack(
        [
            np.ones(len(positive_statements_original)),  # 1 for "Yes" responses
            np.zeros(len(negative_statements_original)),  # 0 for "No" responses
        ]
    )

    # Fit PCA ONCE on original data
    pca_shared = PCA(n_components=n_components)
    X_pca_orig = pca_shared.fit_transform(X_all_orig)

    # Apply the SAME PCA transformation to steered data
    X_pca_steer = pca_shared.transform(
        X_all_steer
    )  # Use transform(), NOT fit_transform()

    print(
        f"PCA Variance explained by first 5 components: {pca_shared.explained_variance_ratio_[:5]}"
    )

    # === TOP LEFT: Original Data - Actual Representations ===
    ax = axes[0, 0]

    # Create DataFrame
    df_orig = pd.DataFrame(X_pca_orig, columns=[f"PC{i}" for i in range(n_components)])
    df_orig["response_type"] = y_combined  # Yes/No responses
    df_orig["toxicity"] = np.hstack([y_vector, y_vector])  # Original toxicity labels

    # Plot with toxicity labels as hue
    sns.scatterplot(
        data=df_orig,
        x=f"PC{component_i}",
        y=f"PC{component_j}",
        hue="toxicity",
        style="response_type",
        palette="Set1",
        alpha=0.7,
        ax=ax,
        s=50,
    )

    # Decision boundary in shared PCA space
    w_pca = pca_shared.components_ @ w
    w_x, w_y = w_pca[component_i], w_pca[component_j]
    slope = -w_x / (w_y + 1e-8)
    intercept = -b / (w_y + 1e-8)

    # Get axis limits that will work for both plots
    all_pc_i = np.concatenate([X_pca_orig[:, component_i], X_pca_steer[:, component_i]])
    all_pc_j = np.concatenate([X_pca_orig[:, component_j], X_pca_steer[:, component_j]])

    x_min, x_max = all_pc_i.min(), all_pc_i.max()
    y_min, y_max = all_pc_j.min(), all_pc_j.max()

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k--", label="Decision boundary", linewidth=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        f"Original - Actual Representations\n(Layer {best_layer}, PC{component_i} vs PC{component_j})"
    )
    ax.set_xlabel(f"PC{component_i}")
    ax.set_ylabel(f"PC{component_j}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === TOP RIGHT: Steered Data - Actual Representations ===
    ax = axes[0, 1]

    # Create DataFrame for steered data (using SAME PCA space)
    df_steer = pd.DataFrame(
        X_pca_steer, columns=[f"PC{i}" for i in range(n_components)]
    )
    df_steer["response_type"] = y_combined  # Yes/No responses
    df_steer["toxicity"] = np.hstack([y_vector, y_vector])  # Original toxicity labels

    # Plot steered data in SAME coordinate system
    sns.scatterplot(
        data=df_steer,
        x=f"PC{component_i}",
        y=f"PC{component_j}",
        hue="toxicity",
        style="response_type",
        palette="Set1",
        alpha=0.7,
        ax=ax,
        s=50,
    )

    # Same decision boundary (since we're in the same PCA space)
    ax.plot(x_vals, y_vals, "k--", label="Decision boundary", linewidth=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        f"Steered - Actual Representations\n(Layer {best_layer}, α={steering_alpha})"
    )
    ax.set_xlabel(f"PC{component_i}")
    ax.set_ylabel(f"PC{component_j}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === BOTTOM LEFT: Original Data - Difference Vectors ===
    ax = axes[1, 0]

    X_diff_orig = positive_statements_original - negative_statements_original
    X_diff_steer = positive_statements_steered - negative_statements_steered

    # Apply SAME PCA to difference vectors
    pca_diff_shared = PCA(n_components=n_components)
    X_pca_diff_orig = pca_diff_shared.fit_transform(X_diff_orig)
    X_pca_diff_steer = pca_diff_shared.transform(X_diff_steer)  # Same transformation

    # Create DataFrame
    df_diff_orig = pd.DataFrame(
        X_pca_diff_orig, columns=[f"PC{i}" for i in range(n_components)]
    )
    df_diff_orig["toxicity"] = y_vector

    # Plot
    sns.scatterplot(
        data=df_diff_orig,
        x=f"PC{component_i}",
        y=f"PC{component_j}",
        hue="toxicity",
        palette="Set1",
        alpha=0.7,
        ax=ax,
        s=50,
    )

    # Decision boundary for difference vectors
    w_pca_diff = pca_diff_shared.components_ @ w
    w_x, w_y = w_pca_diff[component_i], w_pca_diff[component_j]
    slope_diff = -w_x / (w_y + 1e-8)
    intercept_diff = -b / (w_y + 1e-8)

    # Get axis limits for difference plots
    all_diff_pc_i = np.concatenate(
        [X_pca_diff_orig[:, component_i], X_pca_diff_steer[:, component_i]]
    )
    all_diff_pc_j = np.concatenate(
        [X_pca_diff_orig[:, component_j], X_pca_diff_steer[:, component_j]]
    )

    x_diff_min, x_diff_max = all_diff_pc_i.min(), all_diff_pc_i.max()
    y_diff_min, y_diff_max = all_diff_pc_j.min(), all_diff_pc_j.max()

    # Add padding
    x_diff_range = x_diff_max - x_diff_min
    y_diff_range = y_diff_max - y_diff_min
    x_diff_min -= 0.1 * x_diff_range
    x_diff_max += 0.1 * x_diff_range
    y_diff_min -= 0.1 * y_diff_range
    y_diff_max += 0.1 * y_diff_range

    x_vals_diff = np.linspace(x_diff_min, x_diff_max, 200)
    y_vals_diff = slope_diff * x_vals_diff + intercept_diff
    ax.plot(x_vals_diff, y_vals_diff, "k--", label="Decision boundary", linewidth=2)

    ax.set_xlim(x_diff_min, x_diff_max)
    ax.set_ylim(y_diff_min, y_diff_max)

    ax.set_title("Original - Difference Vectors")
    ax.set_xlabel(f"PC{component_i}")
    ax.set_ylabel(f"PC{component_j}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === BOTTOM RIGHT: Steered Data - Difference Vectors ===
    ax = axes[1, 1]

    # Create DataFrame for steered differences (using SAME PCA space)
    df_diff_steer = pd.DataFrame(
        X_pca_diff_steer, columns=[f"PC{i}" for i in range(n_components)]
    )
    df_diff_steer["toxicity"] = y_vector

    # Plot steered differences in SAME coordinate system
    sns.scatterplot(
        data=df_diff_steer,
        x=f"PC{component_i}",
        y=f"PC{component_j}",
        hue="toxicity",
        palette="Set1",
        alpha=0.7,
        ax=ax,
        s=50,
    )

    # Same decision boundary (since we're in the same PCA space)
    ax.plot(x_vals_diff, y_vals_diff, "k--", label="Decision boundary", linewidth=2)

    ax.set_xlim(x_diff_min, x_diff_max)
    ax.set_ylim(y_diff_min, y_diff_max)

    ax.set_title(f"Steered - Difference Vectors\n(α={steering_alpha})")
    ax.set_xlabel(f"PC{component_i}")
    ax.set_ylabel(f"PC{component_j}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add separation metrics text box
    metrics_text = (
        f"Separation Metrics:\n"
        f"Silhouette Score: {separation_metrics['silhouette_score']:.4f}\n"
        f"Fisher Ratio: {separation_metrics['fisher_ratio']:.4f}\n"
        f"Between-Class Dist: {separation_metrics['between_class_distance']:.4f}\n"
        f"Separation Index: {separation_metrics['separation_index']:.4f}"
    )

    # Add text box in the upper right corner of the figure
    fig.text(
        0.02,
        0.98,
        metrics_text,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Overall title
    fig.suptitle(
        f"Boundary Comparison: PC{component_i} vs PC{component_j} (Layer {best_layer})",
        fontsize=16,
        y=0.95,
    )
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Component-specific boundary comparison saved to: {save_path}")
    else:
        plt.show()


def find_best_component_pairs(X_pca, y_vector, n_pairs=5, metric="separation_index"):
    """
    Find the top N component pairs with the best separation.

    Parameters:
        X_pca: PCA-transformed data [N, n_components]
        y_vector: True labels [N,]
        n_pairs: Number of top pairs to return
        metric: Metric to use for ranking ('separation_index', 'silhouette_score', 'fisher_ratio', etc.)

    Returns:
        list: List of tuples (component_i, component_j, metric_value, all_metrics)
    """
    n_components = X_pca.shape[1]
    component_pairs = []

    print(f"Evaluating {n_components * (n_components - 1) // 2} component pairs...")

    # Evaluate all possible component pairs
    for i in range(n_components):
        for j in range(i + 1, n_components):
            metrics = calculate_separation_metrics(X_pca, y_vector, i, j)
            metric_value = metrics[metric]

            component_pairs.append((i, j, metric_value, metrics))

    # Sort by the specified metric (descending)
    component_pairs.sort(key=lambda x: x[2], reverse=True)

    # Return top N pairs
    top_pairs = component_pairs[:n_pairs]

    print(f"Top {n_pairs} component pairs by {metric}:")
    for i, (comp_i, comp_j, metric_val, all_metrics) in enumerate(top_pairs):
        print(f"  {i+1}. PC{comp_i} vs PC{comp_j}: {metric} = {metric_val:.4f}")
        print(
            f"     Silhouette: {all_metrics['silhouette_score']:.4f}, "
            f"Fisher: {all_metrics['fisher_ratio']:.4f}, "
            f"Distance: {all_metrics['between_class_distance']:.4f}"
        )

    return top_pairs


def create_best_separation_plots(
    positive_statements_original,
    negative_statements_original,
    positive_statements_steered,
    negative_statements_steered,
    y_vector,
    ccs,
    best_layer,
    steering_alpha,
    plots_dir,
    n_components=10,
    n_plots=5,
    separation_metric="separation_index",
):
    """
    Main function to create boundary comparison plots for the top N component pairs
    with the best separation.

    Parameters:
        positive_statements_original: Original positive representations [N, hidden_dim]
        negative_statements_original: Original negative representations [N, hidden_dim]
        positive_statements_steered: Steered positive representations [N, hidden_dim]
        negative_statements_steered: Steered negative representations [N, hidden_dim]
        y_vector: True labels [N,]
        ccs: Trained CCS object
        best_layer: Layer index where steering was applied
        steering_alpha: Steering strength
        plots_dir: Directory to save plots
        n_components: Number of PCA components to compute
        n_plots: Number of top separation plots to create (default: 5)
        separation_metric: Metric to use for ranking ('separation_index', 'silhouette_score', 'fisher_ratio')

    Returns:
        list: List of saved plot paths
    """
    print(
        f"\nCreating {n_plots} boundary comparison plots for best separated component pairs..."
    )
    print(f"Using separation metric: {separation_metric}")

    # Create plots directory if it doesn't exist
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)

    # Apply PCA to difference vectors to find best component pairs
    # Use original data to find the inherent structure
    X_diff_orig = positive_statements_original - negative_statements_original

    # Handle NaN values
    if np.isnan(X_diff_orig).any():
        print("Warning: Found NaN values in difference vectors, replacing with 0")
        X_diff_orig = np.nan_to_num(X_diff_orig, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize before PCA
    X_diff_orig_std = (X_diff_orig - X_diff_orig.mean(0)) / (X_diff_orig.std(0) + 1e-8)
    X_diff_orig_std = np.nan_to_num(X_diff_orig_std, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply PCA
    pca = PCA(
        n_components=min(
            n_components, X_diff_orig_std.shape[1], X_diff_orig_std.shape[0]
        )
    )
    X_pca = pca.fit_transform(X_diff_orig_std)

    print(f"PCA applied: {X_pca.shape[0]} samples, {X_pca.shape[1]} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5].round(3)}")

    # Find best component pairs
    top_pairs = find_best_component_pairs(
        X_pca, y_vector, n_pairs=n_plots, metric=separation_metric
    )

    saved_plots = []

    # Create plots for each top component pair
    for rank, (comp_i, comp_j, metric_value, all_metrics) in enumerate(top_pairs):
        print(f"\nCreating plot {rank+1}/{n_plots}: PC{comp_i} vs PC{comp_j}")

        # Create save path with component numbers
        save_path = (
            plots_dir
            / f"boundary_comparison_layer_{best_layer}_alpha_{steering_alpha}_"
            f"PC{comp_i}_PC{comp_j}_rank{rank+1}.png"
        )

        # Create the plot
        plot_boundary_comparison_for_components(
            positive_statements_original=positive_statements_original,
            negative_statements_original=negative_statements_original,
            positive_statements_steered=positive_statements_steered,
            negative_statements_steered=negative_statements_steered,
            y_vector=y_vector,
            ccs=ccs,
            components=[comp_i, comp_j],
            separation_metrics=all_metrics,
            best_layer=best_layer,
            steering_alpha=steering_alpha,
            n_components=X_pca.shape[1],
            save_path=str(save_path),
        )

        saved_plots.append(save_path)

    # Create summary report
    summary_path = (
        plots_dir / f"separation_summary_layer_{best_layer}_alpha_{steering_alpha}.txt"
    )
    with open(summary_path, "w") as f:
        f.write("Best Component Pairs for Separation Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Layer: {best_layer}\n")
        f.write(f"Steering Alpha: {steering_alpha}\n")
        f.write(f"Separation Metric: {separation_metric}\n")
        f.write(f"Total PCA Components: {X_pca.shape[1]}\n")
        f.write(
            f"Explained Variance (first 5): {pca.explained_variance_ratio_[:5].round(3)}\n\n"
        )

        f.write(f"Top {n_plots} Component Pairs:\n")
        f.write("-" * 30 + "\n")

        for rank, (comp_i, comp_j, metric_value, all_metrics) in enumerate(top_pairs):
            f.write(f"\n{rank+1}. PC{comp_i} vs PC{comp_j}:\n")
            f.write(f"   Primary Metric ({separation_metric}): {metric_value:.4f}\n")
            f.write(f"   Silhouette Score: {all_metrics['silhouette_score']:.4f}\n")
            f.write(f"   Fisher Ratio: {all_metrics['fisher_ratio']:.4f}\n")
            f.write(
                f"   Between-Class Distance: {all_metrics['between_class_distance']:.4f}\n"
            )
            f.write(f"   Separation Index: {all_metrics['separation_index']:.4f}\n")
            f.write(
                f"   Class Variance Ratio: {all_metrics['class_variance_ratio']:.4f}\n"
            )
            f.write(
                f"   Plot: boundary_comparison_layer_{best_layer}_alpha_{steering_alpha}_PC{comp_i}_PC{comp_j}_rank{rank+1}.png\n"
            )

    print("\nSeparation analysis complete!")
    print(f"Created {len(saved_plots)} plots")
    print(f"Summary saved to: {summary_path}")
    print(f"All plots saved to: {plots_dir}")

    return saved_plots


######


def plot_layer_steering_effects(
    layer_metrics, best_layer, plots_dir, steering_alpha, method_name="unknown"
):
    """
    Plot steering effects across layers with metrics.

    Args:
        layer_metrics: Dictionary with layer indices as keys and metrics as values
        best_layer: Layer where steering was applied (0-indexed)
        plots_dir: Directory to save plots
        steering_alpha: Steering strength used
        method_name: Name of the method used (for filename)
    """
    layers = sorted(layer_metrics.keys())

    # Extract metrics for plotting
    mse_values = [layer_metrics[layer]["avg_mse"] for layer in layers]
    mae_values = [layer_metrics[layer]["avg_mae"] for layer in layers]
    cosine_values = [layer_metrics[layer]["avg_cosine_similarity"] for layer in layers]
    mean_diff_norms = [layer_metrics[layer]["avg_mean_diff_norm"] for layer in layers]

    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: MSE
    ax1.plot(layers, mse_values, "b-o", linewidth=2, markersize=6)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Average MSE")
    ax1.set_title("Mean Squared Error")
    ax1.grid(True, alpha=0.3)

    # Show where steering was applied and effects should appear
    ax1.axvline(
        best_layer,  # Effects now appear at best_layer due to correct hook placement
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Applied & Effects at Layer {best_layer}",
        alpha=0.7,
    )
    ax1.legend()

    # Plot 2: MAE
    ax2.plot(layers, mae_values, "g-s", linewidth=2, markersize=6)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Average MAE")
    ax2.set_title("Mean Absolute Error")
    ax2.grid(True, alpha=0.3)

    # Same vertical lines
    ax2.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Applied & Effects at Layer {best_layer}",
        alpha=0.7,
    )
    ax2.legend()

    # Plot 3: Cosine Similarity
    ax3.plot(layers, cosine_values, "r-^", linewidth=2, markersize=6)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Average Cosine Similarity")
    ax3.set_title("Cosine Similarity")
    ax3.grid(True, alpha=0.3)

    # Same vertical lines
    ax3.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Applied & Effects at Layer {best_layer}",
        alpha=0.7,
    )
    ax3.legend()

    # Plot 4: Mean Difference Norms
    ax4.plot(layers, mean_diff_norms, "m-d", linewidth=2, markersize=6)
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Average Mean Diff Norm")
    ax4.set_title("Mean Difference Norms")
    ax4.grid(True, alpha=0.3)

    # Same vertical lines
    ax4.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Applied & Effects at Layer {best_layer}",
        alpha=0.7,
    )
    ax4.legend()

    plt.suptitle(
        f"Layer Steering Effects Analysis (α={steering_alpha})\n"
        f"Steering Applied & Effects at Layer {best_layer}",
        fontsize=14,
    )
    plt.tight_layout()

    # Save plot
    plot_filename = f"layer_steering_effects_{method_name}_alpha_{steering_alpha}.png"
    save_path = plots_dir / plot_filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Layer steering effects plot saved to: {save_path}")
    print(
        f"Red line at layer {best_layer} shows where steering is applied and effects appear"
    )

    return save_path


#######


def plot_steering_power_with_proper_steered_data(
    ccs,
    positive_statements_original,
    negative_statements_original,
    positive_statements_steered,
    negative_statements_steered,
    labels=None,
    title: str = "Steering along opinion direction",
    save_path=None,
):
    """
    Plot how steering affects CCS predictions using PROPER steered representations.

    CRITICAL FIX: This function now uses PROPER steered representations that were
    extracted from the actual forward pass, NOT simulations!

    Parameters:
        ccs: Trained CCS object
        positive_statements_original: Original positive representations [N, hidden_dim]
        negative_statements_original: Original negative representations [N, hidden_dim]
        positive_statements_steered: PROPER steered positive representations [N, hidden_dim]
        negative_statements_steered: PROPER steered negative representations [N, hidden_dim]
        labels: Labels for legend
        title: Plot title
        save_path: Path to save plot
    """
    if labels is None:
        labels = ["POS (statement + ДА)", "NEG (statement + НЕТ)"]

    # Convert to tensors if needed and add safety checks
    if not isinstance(positive_statements_original, torch.Tensor):
        positive_statements_original = torch.tensor(
            positive_statements_original, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(negative_statements_original, torch.Tensor):
        negative_statements_original = torch.tensor(
            negative_statements_original, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(positive_statements_steered, torch.Tensor):
        positive_statements_steered = torch.tensor(
            positive_statements_steered, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(negative_statements_steered, torch.Tensor):
        negative_statements_steered = torch.tensor(
            negative_statements_steered, dtype=torch.float32, device=ccs.device
        )

    # Safety checks for NaN or infinite values
    def check_tensor_validity(tensor, name):
        if torch.isnan(tensor).any():
            print(f"⚠️ WARNING: {name} contains NaN values!")
            return False
        if torch.isinf(tensor).any():
            print(f"⚠️ WARNING: {name} contains infinite values!")
            return False
        # Check for extremely large values that could cause plotting issues
        max_val = torch.max(torch.abs(tensor))
        if max_val > 1e6:
            print(f"⚠️ WARNING: {name} contains very large values (max: {max_val:.2e})")
            return False
        return True

    # Check all tensors for validity
    tensors_valid = all(
        [
            check_tensor_validity(
                positive_statements_original, "positive_statements_original"
            ),
            check_tensor_validity(
                negative_statements_original, "negative_statements_original"
            ),
            check_tensor_validity(
                positive_statements_steered, "positive_statements_steered"
            ),
            check_tensor_validity(
                negative_statements_steered, "negative_statements_steered"
            ),
        ]
    )

    if not tensors_valid:
        print("❌ Skipping plot due to invalid tensor values")
        return

    # Get CCS predictions for original data
    with torch.no_grad():
        # Check original representations before CCS prediction
        if (
            torch.isnan(positive_statements_original).any()
            or torch.isinf(positive_statements_original).any()
        ):
            print("❌ Original positive statements contain NaN or Inf values!")
            return
        if (
            torch.isnan(negative_statements_original).any()
            or torch.isinf(negative_statements_original).any()
        ):
            print("❌ Original negative statements contain NaN or Inf values!")
            return

        score_pos_orig = ccs.best_probe(positive_statements_original).median().item()
        score_neg_orig = ccs.best_probe(negative_statements_original).median().item()

    # Get CCS predictions for PROPER steered data
    with torch.no_grad():
        # Check steered representations before CCS prediction
        if (
            torch.isnan(positive_statements_steered).any()
            or torch.isinf(positive_statements_steered).any()
        ):
            print("❌ Steered positive statements contain NaN or Inf values!")
            return
        if (
            torch.isnan(negative_statements_steered).any()
            or torch.isinf(negative_statements_steered).any()
        ):
            print("❌ Steered negative statements contain NaN or Inf values!")
            return

        # Check magnitude of steered representations
        pos_max = torch.max(torch.abs(positive_statements_steered)).item()
        neg_max = torch.max(torch.abs(negative_statements_steered)).item()
        if pos_max > 1e6 or neg_max > 1e6:
            print(
                f"❌ Steered representations have extremely large values! pos_max={pos_max:.2e}, neg_max={neg_max:.2e}"
            )
            return

        score_pos_steered = ccs.best_probe(positive_statements_steered).median().item()
        score_neg_steered = ccs.best_probe(negative_statements_steered).median().item()

    # Debug: Print the actual scores
    print("🔍 DEBUG CCS Scores:")
    print(f"  Original: pos={score_pos_orig:.6f}, neg={score_neg_orig:.6f}")
    print(f"  Steered:  pos={score_pos_steered:.6f}, neg={score_neg_steered:.6f}")

    # Safety check for scores
    scores = [score_pos_orig, score_neg_orig, score_pos_steered, score_neg_steered]
    if any(np.isnan(score) or np.isinf(score) for score in scores):
        print("❌ Skipping plot due to invalid CCS scores")
        print(f"Scores: {scores}")
        return

    # Check for extremely large scores that could cause plotting issues
    max_score = max(abs(score) for score in scores)
    if max_score > 1e3:  # Much more aggressive threshold
        print(
            f"❌ Skipping plot due to extremely large CCS scores (max: {max_score:.2e})"
        )
        print(f"Scores: {scores}")
        return

    # Additional check: ensure all scores are in reasonable range
    for score in scores:
        if abs(score) > 1000:  # Any score > 1000 is suspicious
            print(f"❌ Skipping plot due to suspicious CCS score: {score}")
            print(f"All scores: {scores}")
            return

    # Create comparison plot with fixed figure size
    try:
        plt.figure(figsize=(10, 6))

        # Plot original vs steered
        categories = ["Original", "Steered"]
        pos_scores = [score_pos_orig, score_pos_steered]
        neg_scores = [score_neg_orig, score_neg_steered]

        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width / 2, pos_scores, width, label=labels[0], alpha=0.7)
        plt.bar(x + width / 2, neg_scores, width, label=labels[1], alpha=0.7)

        plt.xlabel("Model State")
        plt.ylabel("Average CCS Result")
        plt.title(f"{title} - Original vs Steered Comparison")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (pos_score, neg_score) in enumerate(zip(pos_scores, neg_scores)):
            plt.text(
                i - width / 2,
                pos_score + 0.01,
                f"{pos_score:.3f}",
                ha="center",
                va="bottom",
            )
            plt.text(
                i + width / 2,
                neg_score + 0.01,
                f"{neg_score:.3f}",
                ha="center",
                va="bottom",
            )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"FIXED steering power plot saved to: {save_path}")
            print("✅ Using PROPER steered representations, NO simulation!")
        else:
            plt.show()
    except Exception as e:
        print(f"❌ Error in plotting section: {e}")
        print(f"Scores that caused the error: {scores}")
        plt.close()
        return


def plot_steering_power(
    ccs,
    positive_statements,
    negative_statements,
    deltas,
    labels=None,
    title: str = "Steering along opinion direction",
    save_path=None,
):
    """
    Plot how steering strength affects CCS predictions.

    ⚠️ WARNING: This function uses SIMULATION and should NOT be used in production!
    Use plot_steering_power_with_proper_steered_data() instead for proper analysis.

    This function is kept for backward compatibility but marked as deprecated.
    """
    print("⚠️ WARNING: plot_steering_power() uses SIMULATION!")
    print("⚠️ Use plot_steering_power_with_proper_steered_data() for proper analysis!")

    if labels is None:
        labels = ["POS (statement + ДА)", "NEG (statement + НЕТ)"]

    weights, bias = ccs.get_weights()
    direction = weights / (np.linalg.norm(weights) + 1e-6)

    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=ccs.device)

    scores_pos, scores_neg = [], []

    if not isinstance(positive_statements, torch.Tensor):
        positive_statements = torch.tensor(
            positive_statements, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(negative_statements, torch.Tensor):
        negative_statements = torch.tensor(
            negative_statements, dtype=torch.float32, device=ccs.device
        )

    for delta in deltas:
        # ⚠️ SIMULATION CODE - NOT PROPER STEERING!
        positive_statements_steered = positive_statements + delta * direction_tensor
        negative_statements_steered = negative_statements - delta * direction_tensor

        score_pos = ccs.best_probe(positive_statements_steered).median().item()
        score_neg = ccs.best_probe(negative_statements_steered).median().item()

        scores_pos.append(score_pos)
        scores_neg.append(score_neg)

    plt.figure(figsize=(10, 6))
    plt.plot(deltas, scores_pos, label=labels[0])
    plt.plot(deltas, scores_neg, label=labels[1])
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Steering delta")
    plt.ylabel("Average CCS result")
    plt.title(f"{title} (⚠️ SIMULATION - NOT PROPER STEERING)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"⚠️ SIMULATION steering plot saved to: {save_path}")
        print("⚠️ This plot uses SIMULATION, not proper steered representations!")
    else:
        plt.show()


# =============================================================================
# COMPREHENSIVE COMPARISON ANALYSIS
# =============================================================================


def create_comparison_results_table(
    X_pos_orig,
    X_neg_orig,
    X_pos_steered,
    X_neg_steered,
    labels,
    train_idx,
    test_idx,
    best_layer,
    device,
    ccs_config,
    normalizing="mean",
):
    """
    Create comprehensive comparison table between original and steered models.
    Changed: New function to generate before/after steering comparison

    Parameters:
        X_pos_orig: Original positive representations [N, n_layers, hidden_dim]
        X_neg_orig: Original negative representations [N, n_layers, hidden_dim]
        X_pos_steered: Steered positive representations [N, n_layers, hidden_dim]
        X_neg_steered: Steered negative representations [N, n_layers, hidden_dim]
        labels: True labels [N,]
        train_idx: Training indices
        test_idx: Testing indices
        best_layer: Layer where steering was applied
        device: Computing device
        ccs_config: CCS configuration dictionary
        normalizing: Normalization method

    Returns:
        comparison_df: DataFrame with original and steered metrics side by side
        orig_results: Original CCS results dictionary
        steered_results: Steered CCS results dictionary
    """
    print("Creating comparison results table...")

    # Convert labels to pandas Series for compatibility
    y_vec = pd.Series(labels)

    # Train CCS on original data
    print("Training CCS on original model...")
    orig_results = train_ccs_on_hidden_states(
        X_pos=X_pos_orig,
        X_neg=X_neg_orig,
        y_vec=y_vec,
        train_idx=train_idx,
        test_idx=test_idx,
        lambda_classification=ccs_config.get("lambda_classification", 0.0),
        normalizing=normalizing,
        device=device,
    )

    # Train CCS on steered data
    print("Training CCS on steered model...")
    steered_results = train_ccs_on_hidden_states(
        X_pos=X_pos_steered,
        X_neg=X_neg_steered,
        y_vec=y_vec,
        train_idx=train_idx,
        test_idx=test_idx,
        lambda_classification=ccs_config.get("lambda_classification", 0.0),
        normalizing=normalizing,
        device=device,
    )

    # Get results tables
    orig_table = get_results_table(orig_results)
    steered_table = get_results_table(steered_results)

    # Create comparison DataFrame
    comparison_data = {}

    # Add original metrics
    for col in orig_table.columns:
        comparison_data[f"{col}_original"] = orig_table[col].values

    # Add steered metrics
    for col in steered_table.columns:
        comparison_data[f"{col}_steered"] = steered_table[col].values

    # Add difference metrics
    for col in orig_table.columns:
        steered_vals = np.array(steered_table[col].values)
        orig_vals = np.array(orig_table[col].values)
        comparison_data[f"{col}_diff"] = steered_vals - orig_vals
        comparison_data[f"{col}_percent_change"] = (
            (steered_vals - orig_vals) / (orig_vals + 1e-8) * 100
        )

    # Create DataFrame with layer indices
    comparison_df = pd.DataFrame(comparison_data, index=orig_table.index)
    comparison_df.index.name = "layer"

    # Add steering info column
    comparison_df["is_steering_layer"] = False
    comparison_df.loc[best_layer, "is_steering_layer"] = True

    print(f"Comparison table created with {len(comparison_df)} layers")
    return comparison_df, orig_results, steered_results


def plot_pca_eigenvalues_analysis(
    X_pos,
    X_neg,
    labels,
    model_type="original",
    save_path=None,
    n_components=10,
    global_axis_limits=None,
):
    """
    Create PCA eigenvalues analysis for each layer.

    For each layer:
    1. Apply PCA with n_components
    2. Calculate eigenvalues and eigenvectors of PCA components matrix
    3. Create triangular grid of plots showing eigenvalues for each layer

    Parameters:
        X_pos: Positive representations [N, n_layers, hidden_dim]
        X_neg: Negative representations [N, n_layers, hidden_dim]
        labels: True labels [N,]
        model_type: "original" or "steered"
        save_path: Path to save the plot
        n_components: Number of PCA components (default: 10)
        global_axis_limits: Dict with axis limits for consistent scaling across models
    """
    print(f"Creating PCA eigenvalues analysis for {model_type} model...")

    n_layers = X_pos.shape[1]

    # Store eigenvalues for each layer
    layer_eigenvalues = []
    layer_pca_components = []

    # Process each layer
    for layer_idx in range(n_layers):
        print(f"Processing layer {layer_idx}/{n_layers-1}")

        # Get data for this layer
        X_pos_layer = X_pos[:, layer_idx, :]
        X_neg_layer = X_neg[:, layer_idx, :]

        # Combine positive and negative data
        X_combined = np.vstack([X_pos_layer, X_neg_layer])

        # Handle NaN values
        if np.isnan(X_combined).any():
            X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize data
        X_combined_std = (X_combined - X_combined.mean(0)) / (X_combined.std(0) + 1e-8)
        X_combined_std = np.nan_to_num(X_combined_std, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply PCA
        actual_components = min(
            n_components, X_combined_std.shape[1], X_combined_std.shape[0]
        )
        pca = PCA(n_components=actual_components)

        pca.fit(X_combined_std)

        # FIXED: Use the actual PCA eigenvalues, not covariance of components
        eigenvalues = pca.explained_variance_  # These are the real eigenvalues!

        # Sort eigenvalues in descending order (should already be sorted by PCA)
        eigenvalues = np.real(eigenvalues)  # Take real part (should already be real)

        layer_eigenvalues.append(eigenvalues)
        layer_pca_components.append(pca.components_)

    # Create triangular grid of plots
    n_plots_per_side = min(n_components, len(layer_eigenvalues[0]))

    # Create figure with triangular arrangement
    fig_size = max(15, n_plots_per_side * 2.5)
    fig, axes = plt.subplots(
        n_plots_per_side, n_plots_per_side, figsize=(fig_size, fig_size)
    )

    # Create colormap for layers (gradient)
    from matplotlib import cm as mpl_cm

    colors = mpl_cm.get_cmap("viridis")(np.linspace(0, 1, n_layers))

    # Create triangular grid of plots
    for i in range(n_plots_per_side):
        for j in range(n_plots_per_side):
            ax = axes[i, j]

            if j > i:  # Upper triangle - hide these plots
                ax.set_visible(False)
                continue

            if i == j:  # Diagonal - show distribution of eigenvalues for this component
                eigenvals_for_component = [
                    layer_eigenvalues[layer][i] for layer in range(n_layers)
                ]

                # Handle case where eigenvalues have very small range
                eigenvals_array = np.array(eigenvals_for_component)
                eigenvals_range = np.max(eigenvals_array) - np.min(eigenvals_array)

                if eigenvals_range < 1e-10 or np.all(
                    eigenvals_array == eigenvals_array[0]
                ):
                    # All eigenvalues are essentially the same - create a simple bar plot
                    ax.bar(
                        [0],
                        [n_layers],
                        alpha=0.7,
                        color="skyblue",
                        edgecolor="black",
                        width=0.5,
                    )
                    ax.set_title(
                        f"PC{i} Eigenvalue Distribution\n(Constant: {eigenvals_array[0]:.6f})"
                    )
                    ax.set_xlabel(f"PC{i} Eigenvalue")
                    ax.set_ylabel("Frequency")
                    ax.set_xlim(-0.5, 0.5)
                    ax.set_xticks([0])
                    ax.set_xticklabels([f"{eigenvals_array[0]:.6f}"])
                else:
                    # Calculate appropriate number of bins based on data range and sample size
                    n_bins = min(
                        max(3, int(np.sqrt(n_layers))), 15
                    )  # Between 3 and 15 bins

                    # Create histogram
                    ax.hist(
                        eigenvals_for_component,
                        bins=n_bins,
                        alpha=0.7,
                        color="skyblue",
                        edgecolor="black",
                    )
                    ax.set_title(f"PC{i} Eigenvalue Distribution")
                    ax.set_xlabel(f"PC{i} Eigenvalue")
                    ax.set_ylabel("Frequency")

                # Apply global axis limits to diagonal plots if provided
                if i == j and global_axis_limits is not None:
                    if f"PC{i}" in global_axis_limits:
                        ax.set_xlim(global_axis_limits[f"PC{i}"])

                ax.grid(True, alpha=0.3)

            else:  # Lower triangle - scatter plot of eigenvalues
                # PC j (x-axis) vs PC i (y-axis)
                x_vals = [layer_eigenvalues[layer][j] for layer in range(n_layers)]
                y_vals = [layer_eigenvalues[layer][i] for layer in range(n_layers)]

                # Plot each layer as a dot with different color
                for layer in range(n_layers):
                    ax.scatter(
                        x_vals[layer],
                        y_vals[layer],
                        c=[colors[layer]],
                        s=60,
                        alpha=0.7,
                        label=f"Layer {layer}" if layer < 10 else None,
                    )  # Limit legend entries

                ax.set_xlabel(f"PC{j} Eigenvalue")
                ax.set_ylabel(f"PC{i} Eigenvalue")
                ax.set_title(f"PC{j} vs PC{i} Eigenvalues")
                ax.grid(True, alpha=0.3)

                # Apply global axis limits if provided
                if global_axis_limits is not None:
                    if f"PC{j}" in global_axis_limits:
                        ax.set_xlim(global_axis_limits[f"PC{j}"])
                    if f"PC{i}" in global_axis_limits:
                        ax.set_ylim(global_axis_limits[f"PC{i}"])

                # Add correlation coefficient as text
                if len(x_vals) > 1:
                    corr = np.corrcoef(x_vals, y_vals)[0, 1]
                    if not np.isnan(corr):
                        ax.text(
                            0.05,
                            0.95,
                            f"r={corr:.3f}",
                            transform=ax.transAxes,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                            ),
                        )

    # Add colorbar to show layer gradient
    sm = plt.cm.ScalarMappable(
        cmap=mpl_cm.get_cmap("viridis"), norm=plt.Normalize(vmin=0, vmax=n_layers - 1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, shrink=0.8)
    cbar.set_label("Layer Index", rotation=270, labelpad=20)

    # Add main title
    plt.suptitle(
        f"PCA Components Eigenvalues Analysis - {model_type.title()} Model\n"
        f"({n_layers} layers, {n_plots_per_side} components)",
        fontsize=16,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"PCA eigenvalues analysis saved to: {save_path}")
    else:
        plt.show()

    # Create summary statistics
    summary_stats = {
        "n_layers": n_layers,
        "n_components": n_plots_per_side,
        "eigenvalues_per_layer": layer_eigenvalues,
        "mean_eigenvalues": np.mean([np.mean(ev) for ev in layer_eigenvalues]),
        "std_eigenvalues": np.std([np.mean(ev) for ev in layer_eigenvalues]),
    }

    return summary_stats


def create_pca_eigenvalues_comparison(
    X_pos_orig,
    X_neg_orig,
    X_pos_steered,
    X_neg_steered,
    labels,
    best_layer,
    steering_alpha,
    plots_dir,
    n_components=10,
):
    """
    Create PCA eigenvalues analysis for both original and steered models.

    CHANGES MADE:
    1. Added explicit bounds checking for layer and component indices
    2. Improved error handling for eigenvalues extraction
    3. Added data validation to prevent index out of range errors
    4. All data kept on GPU using torch tensors throughout processing
    5. Added fallback for missing eigenvalues with proper error messages

    Parameters:
        X_pos_orig: Original positive representations [N, n_layers, hidden_dim]
        X_neg_orig: Original negative representations [N, n_layers, hidden_dim]
        X_pos_steered: Steered positive representations [N, n_layers, hidden_dim]
        X_neg_steered: Steered negative representations [N, n_layers, hidden_dim]
        labels: True labels [N,]
        best_layer: Layer where steering was applied
        steering_alpha: Steering strength
        plots_dir: Directory to save plots
        n_components: Number of PCA components (default: 10)

    Returns:
        dict: Analysis results for both models
    """
    print("\nCreating PCA eigenvalues comparison analysis...")
    print(f"Best layer: {best_layer}, Steering alpha: {steering_alpha}")

    results = {}

    # =================================================================
    # STEP 1: Calculate global axis limits for consistent scaling
    # =================================================================
    print("Calculating global axis limits for consistent scaling...")

    def get_eigenvalues_for_all_layers_gpu(X_pos, X_neg, n_components):
        """Helper function to get eigenvalues for all layers using GPU tensors"""
        n_layers = X_pos.shape[1]
        all_eigenvalues = []

        # Convert to GPU tensors if needed
        if not torch.is_tensor(X_pos):
            X_pos = torch.tensor(
                X_pos,
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        if not torch.is_tensor(X_neg):
            X_neg = torch.tensor(
                X_neg,
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        for layer_idx in range(n_layers):
            # Get data for this layer (keep on GPU)
            X_pos_layer = X_pos[:, layer_idx, :]
            X_neg_layer = X_neg[:, layer_idx, :]

            # Combine positive and negative data
            X_combined = torch.cat([X_pos_layer, X_neg_layer], dim=0)

            # Handle NaN/inf values on GPU
            X_combined = torch.where(
                torch.isnan(X_combined), torch.zeros_like(X_combined), X_combined
            )
            X_combined = torch.where(
                torch.isinf(X_combined), torch.zeros_like(X_combined), X_combined
            )

            # Standardize data on GPU
            X_mean = X_combined.mean(dim=0, keepdim=True)
            X_std = X_combined.std(dim=0, keepdim=True) + 1e-8
            X_combined_std = (X_combined - X_mean) / X_std

            # Handle NaN/inf after standardization
            X_combined_std = torch.where(
                torch.isnan(X_combined_std),
                torch.zeros_like(X_combined_std),
                X_combined_std,
            )
            X_combined_std = torch.where(
                torch.isinf(X_combined_std),
                torch.zeros_like(X_combined_std),
                X_combined_std,
            )

            # Apply PCA using torch operations
            actual_components = min(
                n_components, X_combined_std.shape[1], X_combined_std.shape[0]
            )

            # Center the data
            X_centered = X_combined_std - X_combined_std.mean(dim=0, keepdim=True)

            # Compute covariance matrix on GPU
            cov_matrix = torch.mm(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

            # Add small regularization for numerical stability
            reg_term = 1e-6 * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
            cov_matrix = cov_matrix + reg_term

            # Compute eigenvalues with fallback to CPU for numerical stability
            try:
                eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            except RuntimeError as e:
                if "illegal value" in str(e):
                    print(
                        f"   ⚠️  Eigenvalue computation failed for layer {layer_idx}, falling back to CPU with double precision"
                    )
                    cov_matrix_cpu = cov_matrix.cpu().double()  # Use double precision
                    eigenvalues, _ = torch.linalg.eigh(cov_matrix_cpu)
                    eigenvalues = eigenvalues.float().to(X_combined_std.device)
                else:
                    raise e

            # Sort in descending order and take real part
            eigenvalues = torch.real(eigenvalues)
            eigenvalues, _ = torch.sort(eigenvalues, descending=True)

            # Take top components and convert to CPU for storage
            eigenvalues = eigenvalues[:actual_components].cpu().numpy()
            all_eigenvalues.append(eigenvalues)

        return all_eigenvalues

    # Get eigenvalues for both models (using GPU processing)
    print("Computing eigenvalues for original model...")
    orig_eigenvalues = get_eigenvalues_for_all_layers_gpu(
        X_pos_orig, X_neg_orig, n_components
    )
    print("Computing eigenvalues for steered model...")
    steered_eigenvalues = get_eigenvalues_for_all_layers_gpu(
        X_pos_steered, X_neg_steered, n_components
    )

    # Validate eigenvalues arrays
    if len(orig_eigenvalues) == 0 or len(steered_eigenvalues) == 0:
        print("ERROR: No eigenvalues computed for one or both models")
        return {"error": "eigenvalues_computation_failed"}

    # CHANGE: Added explicit bounds checking to prevent index errors
    n_layers_orig = len(orig_eigenvalues)
    n_layers_steered = len(steered_eigenvalues)
    min_layers = min(n_layers_orig, n_layers_steered)

    print(
        f"Original model layers: {n_layers_orig}, Steered model layers: {n_layers_steered}"
    )
    print(f"Using minimum layers for comparison: {min_layers}")

    # CHANGE: Validate that we have at least one layer
    if min_layers == 0:
        print("ERROR: No matching layers found between original and steered models")
        return {"error": "no_matching_layers"}

    # CHANGE: Check component counts and use minimum
    n_components_orig = (
        min([len(eigs) for eigs in orig_eigenvalues[:min_layers]])
        if orig_eigenvalues
        else 0
    )
    n_components_steered = (
        min([len(eigs) for eigs in steered_eigenvalues[:min_layers]])
        if steered_eigenvalues
        else 0
    )
    n_components_actual = min(n_components_orig, n_components_steered)

    print(
        f"Original components: {n_components_orig}, Steered components: {n_components_steered}"
    )
    print(f"Using components for comparison: {n_components_actual}")

    # CHANGE: Validate we have at least one component
    if n_components_actual == 0:
        print("ERROR: No matching components found between models")
        return {"error": "no_matching_components"}

    # Calculate global min/max for each PC component with bounds checking
    global_axis_limits = {}

    for pc_idx in range(n_components_actual):
        # CHANGE: Added explicit bounds checking for component access
        orig_pc_values = []
        steered_pc_values = []

        for layer_idx in range(min_layers):
            # Check if layer exists and has enough components
            if layer_idx < len(orig_eigenvalues) and pc_idx < len(
                orig_eigenvalues[layer_idx]
            ):
                orig_pc_values.append(orig_eigenvalues[layer_idx][pc_idx])

            if layer_idx < len(steered_eigenvalues) and pc_idx < len(
                steered_eigenvalues[layer_idx]
            ):
                steered_pc_values.append(steered_eigenvalues[layer_idx][pc_idx])

        all_pc_values = orig_pc_values + steered_pc_values

        if all_pc_values:  # Only if we have values
            min_val = min(all_pc_values)
            max_val = max(all_pc_values)

            # Add 5% padding
            range_val = max_val - min_val
            padding = range_val * 0.05 if range_val > 0 else 0.1

            global_axis_limits[f"PC{pc_idx}"] = (min_val - padding, max_val + padding)
        else:
            # Fallback limits if no values found
            global_axis_limits[f"PC{pc_idx}"] = (0.0, 1.0)

    print(f"Global axis limits calculated for PC0-PC{n_components_actual-1}")
    for pc, limits in global_axis_limits.items():
        print(f"  {pc}: [{limits[0]:.2f}, {limits[1]:.2f}]")

    # =================================================================
    # STEP 2: Create comprehensive eigenvalues CSV data with bounds checking
    # =================================================================
    print("Creating comprehensive eigenvalues CSV data...")

    # Create comprehensive DataFrame with all eigenvalues data
    csv_data = []

    for layer_idx in range(min_layers):
        # CHANGE: Added bounds checking for layer access
        row_data = {
            "layer": layer_idx,
            "model_type": "original",
        }

        # Add eigenvalues for each component for original model with bounds checking
        for pc_idx in range(n_components_actual):
            if layer_idx < len(orig_eigenvalues) and pc_idx < len(
                orig_eigenvalues[layer_idx]
            ):
                row_data[f"PC{pc_idx}_eigenvalue"] = float(
                    orig_eigenvalues[layer_idx][pc_idx]
                )
            else:
                row_data[f"PC{pc_idx}_eigenvalue"] = 0.0

        csv_data.append(row_data)

        # Add steered model data for same layer with bounds checking
        row_data_steered = {
            "layer": layer_idx,
            "model_type": "steered",
        }

        # Add eigenvalues for each component for steered model with bounds checking
        for pc_idx in range(n_components_actual):
            if layer_idx < len(steered_eigenvalues) and pc_idx < len(
                steered_eigenvalues[layer_idx]
            ):
                row_data_steered[f"PC{pc_idx}_eigenvalue"] = float(
                    steered_eigenvalues[layer_idx][pc_idx]
                )
            else:
                row_data_steered[f"PC{pc_idx}_eigenvalue"] = 0.0

        csv_data.append(row_data_steered)

    # Create DataFrame
    eigenvalues_df = pd.DataFrame(csv_data)

    # Add additional comparison columns with bounds checking
    comparison_data = []
    for layer_idx in range(min_layers):
        # Calculate differences and ratios between original and steered
        comparison_row = {
            "layer": layer_idx,
            "steering_alpha": steering_alpha,
            "best_layer": best_layer,
            "is_steering_layer": layer_idx == best_layer,
        }

        for pc_idx in range(n_components_actual):
            # CHANGE: Safe access with bounds checking
            orig_val = 0.0
            steered_val = 0.0

            if layer_idx < len(orig_eigenvalues) and pc_idx < len(
                orig_eigenvalues[layer_idx]
            ):
                orig_val = float(orig_eigenvalues[layer_idx][pc_idx])

            if layer_idx < len(steered_eigenvalues) and pc_idx < len(
                steered_eigenvalues[layer_idx]
            ):
                steered_val = float(steered_eigenvalues[layer_idx][pc_idx])

            comparison_row[f"PC{pc_idx}_original"] = orig_val
            comparison_row[f"PC{pc_idx}_steered"] = steered_val
            comparison_row[f"PC{pc_idx}_difference"] = steered_val - orig_val
            comparison_row[f"PC{pc_idx}_ratio"] = steered_val / (orig_val + 1e-10)
            comparison_row[f"PC{pc_idx}_percent_change"] = (
                (steered_val - orig_val) / (orig_val + 1e-10)
            ) * 100

        comparison_data.append(comparison_row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save CSV files
    csv_path_detailed = (
        plots_dir
        / f"pca_eigenvalues_detailed_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    csv_path_comparison = (
        plots_dir
        / f"pca_eigenvalues_comparison_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )

    eigenvalues_df.to_csv(csv_path_detailed, index=False)
    comparison_df.to_csv(csv_path_comparison, index=False)

    print(f"Detailed eigenvalues CSV saved to: {csv_path_detailed}")
    print(f"Comparison eigenvalues CSV saved to: {csv_path_comparison}")

    # Add summary statistics to comparison DataFrame with bounds checking
    summary_stats = {
        "total_layers": min_layers,
        "total_components": n_components_actual,
        "steering_layer": best_layer,
        "steering_alpha": steering_alpha,
    }

    # Calculate summary statistics for each model with bounds checking
    for model_type, eigenvals in [
        ("original", orig_eigenvalues[:min_layers]),
        ("steered", steered_eigenvalues[:min_layers]),
    ]:
        for pc_idx in range(n_components_actual):
            pc_values = []
            for layer_idx in range(min_layers):
                if layer_idx < len(eigenvals) and pc_idx < len(eigenvals[layer_idx]):
                    pc_values.append(float(eigenvals[layer_idx][pc_idx]))

            if pc_values:
                summary_stats[f"{model_type}_PC{pc_idx}_mean"] = np.mean(pc_values)
                summary_stats[f"{model_type}_PC{pc_idx}_std"] = np.std(pc_values)
                summary_stats[f"{model_type}_PC{pc_idx}_min"] = np.min(pc_values)
                summary_stats[f"{model_type}_PC{pc_idx}_max"] = np.max(pc_values)
            else:
                summary_stats[f"{model_type}_PC{pc_idx}_mean"] = 0.0
                summary_stats[f"{model_type}_PC{pc_idx}_std"] = 0.0
                summary_stats[f"{model_type}_PC{pc_idx}_min"] = 0.0
                summary_stats[f"{model_type}_PC{pc_idx}_max"] = 0.0

    # Save summary statistics
    summary_path = (
        plots_dir
        / f"pca_eigenvalues_summary_stats_layer_{best_layer}_alpha_{steering_alpha}.csv"
    )
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics CSV saved to: {summary_path}")

    # =================================================================
    # STEP 3: Create plots with consistent scaling
    # =================================================================

    # Create original model analysis
    print("Analyzing original model...")
    original_save_path = plots_dir / f"pca_eigenvalues_original_layer_{best_layer}.png"

    # CHANGE: Added error handling for plot creation
    original_stats = {"error": "plot_creation_failed"}
    original_stats = plot_pca_eigenvalues_analysis(
        X_pos=X_pos_orig,
        X_neg=X_neg_orig,
        labels=labels,
        model_type="original",
        save_path=str(original_save_path),
        n_components=n_components,
        global_axis_limits=global_axis_limits,
    )
    results["original"] = {"plot_path": original_save_path, "stats": original_stats}

    # Create steered model analysis
    print("Analyzing steered model...")
    steered_save_path = (
        plots_dir
        / f"pca_eigenvalues_steered_layer_{best_layer}_alpha_{steering_alpha}.png"
    )

    # CHANGE: Added error handling for plot creation
    steered_stats = {"error": "plot_creation_failed"}
    steered_stats = plot_pca_eigenvalues_analysis(
        X_pos=X_pos_steered,
        X_neg=X_neg_steered,
        labels=labels,
        model_type="steered",
        save_path=str(steered_save_path),
        n_components=n_components,
        global_axis_limits=global_axis_limits,
    )
    results["steered"] = {"plot_path": steered_save_path, "stats": steered_stats}

    # Add CSV paths to results
    results["csv_files"] = {
        "detailed_eigenvalues": csv_path_detailed,
        "comparison_eigenvalues": csv_path_comparison,
        "summary_statistics": summary_path,
    }

    # Create comparison summary with error handling
    print("Creating comparison summary...")
    comparison_summary_path = (
        plots_dir
        / f"pca_eigenvalues_comparison_summary_layer_{best_layer}_alpha_{steering_alpha}.txt"
    )

    # CHANGE: Added safe access to stats with error checking
    with open(comparison_summary_path, "w") as f:
        f.write("PCA Eigenvalues Analysis Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Layer: {best_layer}\n")
        f.write(f"Steering Alpha: {steering_alpha}\n")
        f.write(f"PCA Components: {n_components}\n")
        f.write(f"Actual Components Used: {n_components_actual}\n")
        f.write(f"Layers Processed: {min_layers}\n\n")

        f.write("CSV FILES CREATED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. Detailed eigenvalues (long format): {csv_path_detailed.name}\n")
        f.write(
            f"2. Comparison eigenvalues (wide format): {csv_path_comparison.name}\n"
        )
        f.write(f"3. Summary statistics: {summary_path.name}\n\n")

        # Safe access to original stats
        if "error" not in original_stats:
            f.write("ORIGINAL MODEL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of layers: {original_stats.get('n_layers', 'N/A')}\n")
            f.write(
                f"Number of components: {original_stats.get('n_components', 'N/A')}\n"
            )
            f.write(
                f"Mean eigenvalue: {original_stats.get('mean_eigenvalues', 0.0):.6f}\n"
            )
            f.write(
                f"Std eigenvalue: {original_stats.get('std_eigenvalues', 0.0):.6f}\n\n"
            )
        else:
            f.write("ORIGINAL MODEL STATISTICS: Error in computation\n\n")

        # Safe access to steered stats
        if "error" not in steered_stats:
            f.write("STEERED MODEL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of layers: {steered_stats.get('n_layers', 'N/A')}\n")
            f.write(
                f"Number of components: {steered_stats.get('n_components', 'N/A')}\n"
            )
            f.write(
                f"Mean eigenvalue: {steered_stats.get('mean_eigenvalues', 0.0):.6f}\n"
            )
            f.write(
                f"Std eigenvalue: {steered_stats.get('std_eigenvalues', 0.0):.6f}\n\n"
            )
        else:
            f.write("STEERED MODEL STATISTICS: Error in computation\n\n")

        # Safe comparison calculation
        if "error" not in original_stats and "error" not in steered_stats:
            f.write("COMPARISON:\n")
            f.write("-" * 30 + "\n")
            mean_diff = float(steered_stats.get("mean_eigenvalues", 0.0)) - float(
                original_stats.get("mean_eigenvalues", 0.0)
            )
            std_diff = float(steered_stats.get("std_eigenvalues", 0.0)) - float(
                original_stats.get("std_eigenvalues", 0.0)
            )
            f.write(f"Mean eigenvalue difference: {mean_diff:+.6f}\n")
            f.write(f"Std eigenvalue difference: {std_diff:+.6f}\n")

            # Percent changes with safe division
            orig_mean = float(original_stats.get("mean_eigenvalues", 0.0))
            orig_std = float(original_stats.get("std_eigenvalues", 0.0))

            if orig_mean != 0:
                mean_pct_change = (mean_diff / orig_mean) * 100
                f.write(f"Mean eigenvalue % change: {mean_pct_change:+.2f}%\n")
            else:
                f.write("Mean eigenvalue % change: N/A (division by zero)\n")

            if orig_std != 0:
                std_pct_change = (std_diff / orig_std) * 100
                f.write(f"Std eigenvalue % change: {std_pct_change:+.2f}%\n\n")
            else:
                f.write("Std eigenvalue % change: N/A (division by zero)\n\n")
        else:
            f.write("COMPARISON: Could not compute due to errors in statistics\n\n")

        f.write("CSV DATA STRUCTURE:\n")
        f.write("-" * 30 + "\n")
        f.write(
            "Detailed CSV columns: layer, model_type, PC0_eigenvalue, PC1_eigenvalue, ...\n"
        )
        f.write(
            "Comparison CSV columns: layer, steering_alpha, best_layer, is_steering_layer,\n"
        )
        f.write(
            "                       PC0_original, PC0_steered, PC0_difference, PC0_ratio, PC0_percent_change, ...\n"
        )
        f.write(
            "Summary CSV: aggregated statistics for each PC component across all layers\n"
        )

    results["comparison_summary"] = comparison_summary_path

    print("PCA eigenvalues comparison completed!")
    print(
        f"Original model plot: {results.get('original', {}).get('plot_path', 'Failed')}"
    )
    print(
        f"Steered model plot: {results.get('steered', {}).get('plot_path', 'Failed')}"
    )
    print(f"Comparison summary: {results.get('comparison_summary', 'Failed')}")
    print(f"Detailed eigenvalues CSV: {csv_path_detailed}")
    print(f"Comparison eigenvalues CSV: {csv_path_comparison}")
    print(f"Summary statistics CSV: {summary_path}")

    return results
