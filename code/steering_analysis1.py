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

    # === TOP LEFT: Original Data - Actual Representations ===
    ax = axes[0, 0]

    # Combine positive and negative representations
    X_all_orig = np.vstack([positive_statements_original, negative_statements_original])
    y_combined = np.hstack(
        [
            np.ones(len(positive_statements_original)),  # 1 for "Yes" responses
            np.zeros(len(negative_statements_original)),  # 0 for "No" responses
        ]
    )

    # Apply PCA
    pca_orig = PCA(n_components=n_components)
    X_pca_orig = pca_orig.fit_transform(X_all_orig)

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

    # Decision boundary in PCA space
    w_pca_orig = pca_orig.components_ @ w
    w_x, w_y = w_pca_orig[component_i], w_pca_orig[component_j]
    slope = -w_x / (w_y + 1e-8)
    intercept = -b / (w_y + 1e-8)

    x_vals = np.linspace(
        df_orig[f"PC{component_i}"].min(), df_orig[f"PC{component_i}"].max(), 200
    )
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k--", label="Decision boundary", linewidth=2)

    ax.set_title(
        f"Original - Actual Representations\n(Layer {best_layer}, PC{component_i} vs PC{component_j})"
    )
    ax.set_xlabel(f"PC{component_i}")
    ax.set_ylabel(f"PC{component_j}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === TOP RIGHT: Steered Data - Actual Representations ===
    ax = axes[0, 1]

    # Combine steered positive and negative representations
    X_all_steer = np.vstack([positive_statements_steered, negative_statements_steered])

    # Apply PCA (fit on steered data)
    pca_steer = PCA(n_components=n_components)
    X_pca_steer = pca_steer.fit_transform(X_all_steer)

    # Create DataFrame
    df_steer = pd.DataFrame(
        X_pca_steer, columns=[f"PC{i}" for i in range(n_components)]
    )
    df_steer["response_type"] = y_combined  # Yes/No responses
    df_steer["toxicity"] = np.hstack([y_vector, y_vector])  # Original toxicity labels

    # Plot with toxicity labels as hue
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

    # Decision boundary in steered PCA space
    w_pca_steer = pca_steer.components_ @ w
    w_x, w_y = w_pca_steer[component_i], w_pca_steer[component_j]
    slope = -w_x / (w_y + 1e-8)
    intercept = -b / (w_y + 1e-8)

    x_vals = np.linspace(
        df_steer[f"PC{component_i}"].min(), df_steer[f"PC{component_i}"].max(), 200
    )
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k--", label="Decision boundary", linewidth=2)

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

    # Apply PCA to difference vectors
    pca_diff_orig = PCA(n_components=n_components)
    X_pca_diff_orig = pca_diff_orig.fit_transform(X_diff_orig)

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
    w_pca_diff_orig = pca_diff_orig.components_ @ w
    w_x, w_y = w_pca_diff_orig[component_i], w_pca_diff_orig[component_j]
    slope = -w_x / (w_y + 1e-8)
    intercept = -b / (w_y + 1e-8)

    x_vals = np.linspace(
        df_diff_orig[f"PC{component_i}"].min(),
        df_diff_orig[f"PC{component_i}"].max(),
        200,
    )
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k--", label="Decision boundary", linewidth=2)

    ax.set_title("Original - Difference Vectors")
    ax.set_xlabel(f"PC{component_i}")
    ax.set_ylabel(f"PC{component_j}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === BOTTOM RIGHT: Steered Data - Difference Vectors ===
    ax = axes[1, 1]

    X_diff_steer = positive_statements_steered - negative_statements_steered

    # Apply PCA to steered difference vectors
    pca_diff_steer = PCA(n_components=n_components)
    X_pca_diff_steer = pca_diff_steer.fit_transform(X_diff_steer)

    # Create DataFrame
    df_diff_steer = pd.DataFrame(
        X_pca_diff_steer, columns=[f"PC{i}" for i in range(n_components)]
    )
    df_diff_steer["toxicity"] = y_vector

    # Plot
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

    # Decision boundary for steered difference vectors
    w_pca_diff_steer = pca_diff_steer.components_ @ w
    w_x, w_y = w_pca_diff_steer[component_i], w_pca_diff_steer[component_j]
    slope = -w_x / (w_y + 1e-8)
    intercept = -b / (w_y + 1e-8)

    x_vals = np.linspace(
        df_diff_steer[f"PC{component_i}"].min(),
        df_diff_steer[f"PC{component_i}"].max(),
        200,
    )
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k--", label="Decision boundary", linewidth=2)

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


def plot_layer_steering_effects(layer_metrics, best_layer, plots_dir, steering_alpha):
    """
    Plot quantitative steering effects across layers.

    Parameters:
        layer_metrics: Dict from compare_steering_layers function
        best_layer: Layer where steering was applied
        plots_dir: Directory to save plots
        steering_alpha: Steering strength used
    """
    layers = sorted(layer_metrics.keys())

    # Extract metrics for plotting
    metrics_to_plot = [
        "avg_mse",
        "avg_mae",
        "avg_cosine_similarity",
        "avg_mean_diff_norm",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        values = [layer_metrics[layer][metric] for layer in layers]

        ax.plot(layers, values, "b-o", linewidth=2, markersize=6)
        ax.axvline(
            best_layer,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Steering Layer {best_layer}",
        )

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.replace("avg_", "").replace("_", " ").title())
        ax.set_title(
            f'{metric.replace("avg_", "").replace("_", " ").title()} Across Layers'
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(f"Steering Effects Across Layers (α={steering_alpha})", fontsize=16)
    plt.tight_layout()

    save_path = plots_dir / f"layer_steering_effects_alpha_{steering_alpha}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Layer steering effects plot saved to: {save_path}")
    return save_path


#######


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

    Changed: Added save_path parameter to save plot instead of showing
    """
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
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Steering plot saved to: {save_path}")
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
