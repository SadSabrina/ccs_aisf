"""
FIXED Steering Analysis and Visualization Functions

This module contains FIXED analysis and plotting functions for steering experiments.
KEY FIX: All PCA transformations use the SAME coordinate system for original and steered data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Import CCS functions
from sklearn.decomposition import PCA

# =============================================================================
# COMPREHENSIVE COMPARISON VISUALIZATION FUNCTIONS
# =============================================================================


def plot_steering_layer_analysis(layer_metrics, best_layer, save_path):
    """
    Create steering layer analysis plot (bar chart showing effects across layers).
    Restored: This function creates the steering_layer_analysis.png plot.
    """
    layers = sorted(layer_metrics.keys())

    # Extract MSE values for plotting
    mse_values = [layer_metrics[layer]["avg_mse"] for layer in layers]
    mae_values = [layer_metrics[layer]["avg_mae"] for layer in layers]
    cosine_values = [layer_metrics[layer]["avg_cosine_similarity"] for layer in layers]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # MSE plot
    bars1 = ax1.bar(layers, mse_values, alpha=0.7, color="skyblue")
    if best_layer < len(bars1):
        bars1[best_layer].set_color("red")  # Highlight steering layer
    ax1.axvline(
        best_layer,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Steering Applied at Layer {best_layer}",
    )
    if best_layer + 1 < len(layers):
        ax1.axvline(
            best_layer + 1,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"Effects Begin at Layer {best_layer + 1}",
        )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Average MSE")
    ax1.set_title("Mean Squared Error Across Layers")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # MAE plot
    bars2 = ax2.bar(layers, mae_values, alpha=0.7, color="lightgreen")
    if best_layer < len(bars2):
        bars2[best_layer].set_color("red")  # Highlight steering layer
    ax2.axvline(
        best_layer,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Steering Applied at Layer {best_layer}",
    )
    if best_layer + 1 < len(layers):
        ax2.axvline(
            best_layer + 1,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"Effects Begin at Layer {best_layer + 1}",
        )
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Average MAE")
    ax2.set_title("Mean Absolute Error Across Layers")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Cosine similarity plot
    bars3 = ax3.bar(layers, cosine_values, alpha=0.7, color="lightcoral")
    if best_layer < len(bars3):
        bars3[best_layer].set_color("red")  # Highlight steering layer
    ax3.axvline(
        best_layer,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Steering Applied at Layer {best_layer}",
    )
    if best_layer + 1 < len(layers):
        ax3.axvline(
            best_layer + 1,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"Effects Begin at Layer {best_layer + 1}",
        )
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Average Cosine Similarity")
    ax3.set_title("Cosine Similarity Across Layers")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.suptitle(
        f"Steering Effects Analysis (Steering Layer: {best_layer})", fontsize=16
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Steering layer analysis saved to: {save_path}")


def plot_improved_layerwise_steering_focus(
    X_pos,
    X_neg,
    X_pos_steered,
    X_neg_steered,
    labels,
    best_layer,
    steering_alpha,
    save_path,
):
    """
    Create improved layerwise steering focus plot.
    Restored: This function creates the improved_layerwise_steering_focus_layer_X.png plot.
    """

    n_layers = X_pos.shape[1]

    # Focus on layers around the steering layer (±3 layers)
    start_layer = max(0, best_layer - 3)
    end_layer = min(n_layers, best_layer + 4)
    focus_layers = list(range(start_layer, end_layer))

    n_focus = len(focus_layers)
    fig, axes = plt.subplots(2, n_focus, figsize=(4 * n_focus, 8))

    if n_focus == 1:
        axes = axes.reshape(2, 1)

    for i, layer_idx in enumerate(focus_layers):
        # === CRITICAL FIX: Use SAME PCA for original and steered data ===

        # Get difference vectors
        X_diff_orig = X_pos[:, layer_idx, :] - X_neg[:, layer_idx, :]
        X_diff_steer = X_pos_steered[:, layer_idx, :] - X_neg_steered[:, layer_idx, :]

        # Handle NaN values
        if np.isnan(X_diff_orig).any():
            X_diff_orig = np.nan_to_num(X_diff_orig, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(X_diff_steer).any():
            X_diff_steer = np.nan_to_num(X_diff_steer, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize ORIGINAL data
        X_diff_orig_std = (X_diff_orig - X_diff_orig.mean(0)) / (
            X_diff_orig.std(0) + 1e-8
        )
        X_diff_orig_std = np.nan_to_num(
            X_diff_orig_std, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Standardize STEERED data using ORIGINAL statistics
        X_diff_steer_std = (X_diff_steer - X_diff_orig.mean(0)) / (
            X_diff_orig.std(0) + 1e-8
        )
        X_diff_steer_std = np.nan_to_num(
            X_diff_steer_std, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Fit PCA ONCE on original data
        pca_shared = PCA(n_components=2)
        X_pca_orig = pca_shared.fit_transform(X_diff_orig_std)

        # Apply SAME PCA transformation to steered data
        X_pca_steer = pca_shared.transform(X_diff_steer_std)

        # Get axis limits that work for both plots
        all_pc1 = np.concatenate([X_pca_orig[:, 0], X_pca_steer[:, 0]])
        all_pc2 = np.concatenate([X_pca_orig[:, 1], X_pca_steer[:, 1]])

        x_min, x_max = all_pc1.min(), all_pc1.max()
        y_min, y_max = all_pc2.min(), all_pc2.max()

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # Original data plot
        ax_orig = axes[0, i]
        scatter_orig = ax_orig.scatter(
            X_pca_orig[:, 0], X_pca_orig[:, 1], c=labels, cmap="Set1", alpha=0.7, s=30
        )
        ax_orig.set_xlim(x_min, x_max)
        ax_orig.set_ylim(y_min, y_max)
        ax_orig.set_title(
            f"Original Layer {layer_idx}"
            + (" (STEERING)" if layer_idx == best_layer else "")
        )
        ax_orig.set_xlabel("PC1")
        ax_orig.set_ylabel("PC2")
        ax_orig.grid(True, alpha=0.3)

        # Steered data plot (SAME coordinate system)
        ax_steer = axes[1, i]
        scatter_steer = ax_steer.scatter(
            X_pca_steer[:, 0], X_pca_steer[:, 1], c=labels, cmap="Set1", alpha=0.7, s=30
        )
        ax_steer.set_xlim(x_min, x_max)
        ax_steer.set_ylim(y_min, y_max)
        ax_steer.set_title(
            f"Steered Layer {layer_idx} (α={steering_alpha})"
            + (" (STEERING)" if layer_idx == best_layer else "")
        )
        ax_steer.set_xlabel("PC1")
        ax_steer.set_ylabel("PC2")
        ax_steer.grid(True, alpha=0.3)

        # Calculate and display movement statistics for steering layer
        if layer_idx == best_layer:
            movement = X_pca_steer - X_pca_orig
            mean_movement_x = np.mean(movement[:, 0])
            mean_movement_y = np.mean(movement[:, 1])
            total_movement = np.mean(np.linalg.norm(movement, axis=1))

            print(f"Layer {layer_idx} movement stats:")
            print(f"  Mean PC1 movement: {mean_movement_x:.4f}")
            print(f"  Mean PC2 movement: {mean_movement_y:.4f}")
            print(f"  Total movement magnitude: {total_movement:.4f}")

    plt.suptitle(
        f"FIXED Layerwise Steering Analysis (Steering Layer: {best_layer})", fontsize=16
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Improved layerwise comparison saved to: {save_path}")


def plot_steering_power_improved_with_proper_steered_data(
    ccs,
    X_pos_original,
    X_neg_original,
    X_pos_steered,
    X_neg_steered,
    best_layer,
    steering_alpha,
    save_path,
):
    """
    Create improved steering power plot using PROPER steered representations.

    CRITICAL FIX: This function now uses PROPER steered representations that were
    extracted from the actual forward pass, NOT simulations!

    Parameters:
        ccs: Trained CCS object
        X_pos_original: Original positive representations [N, hidden_dim]
        X_neg_original: Original negative representations [N, hidden_dim]
        X_pos_steered: PROPER steered positive representations [N, hidden_dim]
        X_neg_steered: PROPER steered negative representations [N, hidden_dim]
        best_layer: Layer where steering was applied
        steering_alpha: Steering strength used
        save_path: Path to save plot
    """
    print("Creating improved steering power plot with PROPER steered data...")

    # Convert to tensors if needed
    if not isinstance(X_pos_original, torch.Tensor):
        X_pos_original = torch.tensor(
            X_pos_original, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(X_neg_original, torch.Tensor):
        X_neg_original = torch.tensor(
            X_neg_original, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(X_pos_steered, torch.Tensor):
        X_pos_steered = torch.tensor(
            X_pos_steered, dtype=torch.float32, device=ccs.device
        )
    if not isinstance(X_neg_steered, torch.Tensor):
        X_neg_steered = torch.tensor(
            X_neg_steered, dtype=torch.float32, device=ccs.device
        )

    # Get CCS predictions for original data
    with torch.no_grad():
        pos_pred_orig = ccs.best_probe(X_pos_original).median().item()
        neg_pred_orig = ccs.best_probe(X_neg_original).median().item()

    # Get CCS predictions for PROPER steered data
    with torch.no_grad():
        pos_pred_steered = ccs.best_probe(X_pos_steered).median().item()
        neg_pred_steered = ccs.best_probe(X_neg_steered).median().item()

    # Calculate metrics
    separation_orig = abs(pos_pred_orig - neg_pred_orig)
    separation_steered = abs(pos_pred_steered - neg_pred_steered)

    confidence_orig = 0.5 * (pos_pred_orig + (1 - neg_pred_orig))
    confidence_steered = 0.5 * (pos_pred_steered + (1 - neg_pred_steered))

    # Create enhanced comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Basic steering comparison
    categories = ["Original", "Steered"]
    pos_scores = [pos_pred_orig, pos_pred_steered]
    neg_scores = [neg_pred_orig, neg_pred_steered]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(
        x - width / 2,
        pos_scores,
        width,
        label="Positive Statements",
        alpha=0.7,
        color="blue",
    )
    ax1.bar(
        x + width / 2,
        neg_scores,
        width,
        label="Negative Statements",
        alpha=0.7,
        color="red",
    )
    ax1.set_xlabel("Model State")
    ax1.set_ylabel("CCS Prediction")
    ax1.set_title("CCS Predictions: Original vs Steered")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (pos_score, neg_score) in enumerate(zip(pos_scores, neg_scores)):
        ax1.text(
            i - width / 2,
            pos_score + 0.01,
            f"{pos_score:.3f}",
            ha="center",
            va="bottom",
        )
        ax1.text(
            i + width / 2,
            neg_score + 0.01,
            f"{neg_score:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 2: Separation comparison
    separations = [separation_orig, separation_steered]
    bars = ax2.bar(categories, separations, alpha=0.7, color=["green", "orange"])
    ax2.set_xlabel("Model State")
    ax2.set_ylabel("Prediction Separation")
    ax2.set_title("Prediction Separation: Original vs Steered")
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, sep in zip(bars, separations):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{sep:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 3: Confidence comparison
    confidences = [confidence_orig, confidence_steered]
    bars = ax3.bar(categories, confidences, alpha=0.7, color=["purple", "brown"])
    ax3.set_xlabel("Model State")
    ax3.set_ylabel("Average Confidence")
    ax3.set_title("Average Confidence: Original vs Steered")
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{conf:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 4: Combined metrics comparison
    metrics = ["Separation", "Confidence"]
    orig_metrics = [separation_orig, confidence_orig]
    steered_metrics = [separation_steered, confidence_steered]

    x = np.arange(len(metrics))
    width = 0.35

    ax4.bar(
        x - width / 2, orig_metrics, width, label="Original", alpha=0.7, color="cyan"
    )
    ax4.bar(
        x + width / 2,
        steered_metrics,
        width,
        label="Steered",
        alpha=0.7,
        color="magenta",
    )
    ax4.set_xlabel("Metrics")
    ax4.set_ylabel("Value")
    ax4.set_title("Combined Metrics Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for i, (orig_val, steered_val) in enumerate(zip(orig_metrics, steered_metrics)):
        ax4.text(
            i - width / 2, orig_val + 0.01, f"{orig_val:.3f}", ha="center", va="bottom"
        )
        ax4.text(
            i + width / 2,
            steered_val + 0.01,
            f"{steered_val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.suptitle(
        f"FIXED Steering Power Analysis - Layer {best_layer} (α={steering_alpha})",
        fontsize=16,
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"FIXED steering power plot saved to: {save_path}")
    print("✅ Using PROPER steered representations, NO simulation!")

    # Print summary of changes
    print("📊 Steering Effects Summary:")
    print(
        f"   Positive predictions: {pos_pred_orig:.3f} → {pos_pred_steered:.3f} (Δ={pos_pred_steered-pos_pred_orig:+.3f})"
    )
    print(
        f"   Negative predictions: {neg_pred_orig:.3f} → {neg_pred_steered:.3f} (Δ={neg_pred_steered-neg_pred_orig:+.3f})"
    )
    print(
        f"   Separation: {separation_orig:.3f} → {separation_steered:.3f} (Δ={separation_steered-separation_orig:+.3f})"
    )
    print(
        f"   Confidence: {confidence_orig:.3f} → {confidence_steered:.3f} (Δ={confidence_steered-confidence_orig:+.3f})"
    )


def plot_steering_power_improved(
    ccs, X_pos, X_neg, direction_tensor, best_layer, save_path
):
    """
    Create improved steering power plot with additional metrics.

    ⚠️ WARNING: This function uses SIMULATION and should NOT be used in production!
    Use plot_steering_power_improved_with_proper_steered_data() instead for proper analysis.

    This function is kept for backward compatibility but marked as deprecated.
    """
    print("⚠️ WARNING: plot_steering_power_improved() uses SIMULATION!")
    print(
        "⚠️ Use plot_steering_power_improved_with_proper_steered_data() for proper analysis!"
    )

    deltas = np.linspace(-3, 3, 21)

    # Convert to tensors
    if not isinstance(X_pos, torch.Tensor):
        X_pos_tensor = torch.tensor(X_pos, dtype=torch.float32, device=ccs.device)
    else:
        X_pos_tensor = X_pos

    if not isinstance(X_neg, torch.Tensor):
        X_neg_tensor = torch.tensor(X_neg, dtype=torch.float32, device=ccs.device)
    else:
        X_neg_tensor = X_neg

    # Get direction
    direction_np = (
        direction_tensor.cpu().numpy()
        if torch.is_tensor(direction_tensor)
        else direction_tensor
    )
    direction_torch = torch.tensor(direction_np, dtype=torch.float32, device=ccs.device)

    pos_scores = []
    neg_scores = []
    separations = []
    confidences = []

    for delta in deltas:
        # ⚠️ SIMULATION CODE - NOT PROPER STEERING!
        X_pos_steered = X_pos_tensor + delta * direction_torch
        X_neg_steered = X_neg_tensor - delta * direction_torch

        # Get CCS predictions
        with torch.no_grad():
            pos_pred = ccs.best_probe(X_pos_steered).median().item()
            neg_pred = ccs.best_probe(X_neg_steered).median().item()

        pos_scores.append(pos_pred)
        neg_scores.append(neg_pred)

        # Calculate separation
        separation = abs(pos_pred - neg_pred)
        separations.append(separation)

        # Calculate confidence
        confidence = 0.5 * (pos_pred + (1 - neg_pred))
        confidences.append(confidence)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Basic steering power
    ax1.plot(deltas, pos_scores, "b-o", label="Positive Statements", linewidth=2)
    ax1.plot(deltas, neg_scores, "r-s", label="Negative Statements", linewidth=2)
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Steering Delta")
    ax1.set_ylabel("CCS Prediction")
    ax1.set_title("CCS Predictions vs Steering Strength (⚠️ SIMULATION)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Separation
    ax2.plot(deltas, separations, "g-^", linewidth=2, markersize=6)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Steering Delta")
    ax2.set_ylabel("Prediction Separation")
    ax2.set_title("Prediction Separation vs Steering (⚠️ SIMULATION)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Confidence
    ax3.plot(deltas, confidences, "m-d", linewidth=2, markersize=6)
    ax3.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Steering Delta")
    ax3.set_ylabel("Average Confidence")
    ax3.set_title("Average Confidence vs Steering (⚠️ SIMULATION)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Combined view
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(deltas, separations, "g-^", label="Separation", linewidth=2)
    line2 = ax4_twin.plot(deltas, confidences, "m-d", label="Confidence", linewidth=2)
    ax4.axvline(0, color="gray", linestyle="--", alpha=0.7)

    ax4.set_xlabel("Steering Delta")
    ax4.set_ylabel("Separation", color="g")
    ax4_twin.set_ylabel("Confidence", color="m")
    ax4.set_title("Separation & Confidence vs Steering (⚠️ SIMULATION)")

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper left")

    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"⚠️ SIMULATION Steering Power Analysis - Layer {best_layer}", fontsize=16
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"⚠️ SIMULATION steering plot saved to: {save_path}")
    print("⚠️ This plot uses SIMULATION, not proper steered representations!")


def plot_boundary_comparison_improved(
    X_pos_orig,
    X_neg_orig,
    X_pos_steer,
    X_neg_steer,
    labels,
    ccs,
    best_layer,
    steering_alpha,
    save_path,
):
    """
    FIXED: Create improved boundary comparison plot using SAME PCA coordinates.

    KEY FIX: Uses the same PCA transformation for original and steered data,
    so you can see how steering moved the data in the same coordinate system.
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Get CCS weights for decision boundary
    weights, bias = ccs.get_weights()

    # === CRITICAL FIX: Use SAME PCA for all plots ===

    # Get difference vectors for both original and steered
    X_diff_orig = X_pos_orig - X_neg_orig
    X_diff_steer = X_pos_steer - X_neg_steer

    # Handle NaN values
    if np.isnan(X_diff_orig).any():
        X_diff_orig = np.nan_to_num(X_diff_orig, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(X_diff_steer).any():
        X_diff_steer = np.nan_to_num(X_diff_steer, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize based on original data
    X_diff_orig_std = (X_diff_orig - X_diff_orig.mean(0)) / (X_diff_orig.std(0) + 1e-8)
    X_diff_orig_std = np.nan_to_num(X_diff_orig_std, nan=0.0, posinf=0.0, neginf=0.0)

    X_diff_steer_std = (X_diff_steer - X_diff_orig.mean(0)) / (
        X_diff_orig.std(0) + 1e-8
    )
    X_diff_steer_std = np.nan_to_num(X_diff_steer_std, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit PCA ONCE on original difference vectors
    pca_diff_shared = PCA(n_components=2)
    X_pca_diff_orig = pca_diff_shared.fit_transform(X_diff_orig_std)

    # Apply SAME PCA transformation to steered difference vectors
    X_pca_diff_steer = pca_diff_shared.transform(X_diff_steer_std)

    # Get axis limits for both difference plots
    all_diff_pc1 = np.concatenate([X_pca_diff_orig[:, 0], X_pca_diff_steer[:, 0]])
    all_diff_pc2 = np.concatenate([X_pca_diff_orig[:, 1], X_pca_diff_steer[:, 1]])

    x_diff_min, x_diff_max = all_diff_pc1.min(), all_diff_pc1.max()
    y_diff_min, y_diff_max = all_diff_pc2.min(), all_diff_pc2.max()

    # Add padding
    x_diff_range = x_diff_max - x_diff_min
    y_diff_range = y_diff_max - y_diff_min
    x_diff_min -= 0.1 * x_diff_range
    x_diff_max += 0.1 * x_diff_range
    y_diff_min -= 0.1 * y_diff_range
    y_diff_max += 0.1 * y_diff_range

    # Plot 1: Original difference vectors
    ax = axes[0, 0]
    scatter_orig = ax.scatter(
        X_pca_diff_orig[:, 0],
        X_pca_diff_orig[:, 1],
        c=labels,
        cmap="Set1",
        alpha=0.7,
        s=50,
    )
    ax.set_xlim(x_diff_min, x_diff_max)
    ax.set_ylim(y_diff_min, y_diff_max)

    # Decision boundary for difference vectors
    w_pca_diff = pca_diff_shared.components_ @ weights
    if abs(w_pca_diff[1]) > 1e-8:
        slope_diff = -w_pca_diff[0] / w_pca_diff[1]
        intercept_diff = -bias / w_pca_diff[1]
        x_vals_diff = np.linspace(x_diff_min, x_diff_max, 100)
        y_vals_diff = slope_diff * x_vals_diff + intercept_diff
        ax.plot(x_vals_diff, y_vals_diff, "k--", label="Decision boundary", linewidth=2)
        ax.legend()

    ax.set_title(f"Original - Difference Vectors (Layer {best_layer})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # Plot 2: Steered difference vectors (SAME coordinate system)
    ax = axes[0, 1]
    scatter_steer = ax.scatter(
        X_pca_diff_steer[:, 0],
        X_pca_diff_steer[:, 1],
        c=labels,
        cmap="Set1",
        alpha=0.7,
        s=50,
    )
    ax.set_xlim(x_diff_min, x_diff_max)
    ax.set_ylim(y_diff_min, y_diff_max)

    # Same decision boundary
    if abs(w_pca_diff[1]) > 1e-8:
        ax.plot(x_vals_diff, y_vals_diff, "k--", label="Decision boundary", linewidth=2)
        ax.legend()

    ax.set_title(f"Steered - Difference Vectors (α={steering_alpha})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # === Combined representations - also use SAME PCA ===

    # Get combined data for both original and steered
    X_combined_orig = np.vstack([X_pos_orig, X_neg_orig])
    X_combined_steer = np.vstack([X_pos_steer, X_neg_steer])
    y_combined = np.hstack([np.ones(len(X_pos_orig)), np.zeros(len(X_neg_orig))])
    labels_combined = np.hstack([labels, labels])

    # Handle NaN values
    if np.isnan(X_combined_orig).any():
        X_combined_orig = np.nan_to_num(
            X_combined_orig, nan=0.0, posinf=0.0, neginf=0.0
        )
    if np.isnan(X_combined_steer).any():
        X_combined_steer = np.nan_to_num(
            X_combined_steer, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Standardize based on original data
    X_combined_orig_std = (X_combined_orig - X_combined_orig.mean(0)) / (
        X_combined_orig.std(0) + 1e-8
    )
    X_combined_orig_std = np.nan_to_num(
        X_combined_orig_std, nan=0.0, posinf=0.0, neginf=0.0
    )

    X_combined_steer_std = (X_combined_steer - X_combined_orig.mean(0)) / (
        X_combined_orig.std(0) + 1e-8
    )
    X_combined_steer_std = np.nan_to_num(
        X_combined_steer_std, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Fit PCA ONCE on original combined data
    pca_combined_shared = PCA(n_components=2)
    X_pca_combined_orig = pca_combined_shared.fit_transform(X_combined_orig_std)

    # Apply SAME PCA transformation to steered combined data
    X_pca_combined_steer = pca_combined_shared.transform(X_combined_steer_std)

    # Get axis limits for both combined plots
    all_combined_pc1 = np.concatenate(
        [X_pca_combined_orig[:, 0], X_pca_combined_steer[:, 0]]
    )
    all_combined_pc2 = np.concatenate(
        [X_pca_combined_orig[:, 1], X_pca_combined_steer[:, 1]]
    )

    x_comb_min, x_comb_max = all_combined_pc1.min(), all_combined_pc1.max()
    y_comb_min, y_comb_max = all_combined_pc2.min(), all_combined_pc2.max()

    # Add padding
    x_comb_range = x_comb_max - x_comb_min
    y_comb_range = y_comb_max - y_comb_min
    x_comb_min -= 0.1 * x_comb_range
    x_comb_max += 0.1 * x_comb_range
    y_comb_min -= 0.1 * y_comb_range
    y_comb_max += 0.1 * y_comb_range

    # Plot 3: Original combined representations
    ax = axes[1, 0]

    # Create DataFrame for plotting
    df_combined = pd.DataFrame(
        {
            "PC1": X_pca_combined_orig[:, 0],
            "PC2": X_pca_combined_orig[:, 1],
            "response": y_combined,
            "toxicity": labels_combined,
        }
    )

    sns.scatterplot(
        data=df_combined,
        x="PC1",
        y="PC2",
        hue="toxicity",
        style="response",
        ax=ax,
        alpha=0.7,
        s=50,
    )
    ax.set_xlim(x_comb_min, x_comb_max)
    ax.set_ylim(y_comb_min, y_comb_max)
    ax.set_title("Original - Combined Representations")
    ax.grid(True, alpha=0.3)

    # Decision boundary for combined data
    w_pca_combined = pca_combined_shared.components_ @ weights
    if abs(w_pca_combined[1]) > 1e-8:
        slope_combined = -w_pca_combined[0] / w_pca_combined[1]
        intercept_combined = -bias / w_pca_combined[1]
        x_vals_combined = np.linspace(x_comb_min, x_comb_max, 100)
        y_vals_combined = slope_combined * x_vals_combined + intercept_combined
        ax.plot(
            x_vals_combined,
            y_vals_combined,
            "k--",
            label="Decision boundary",
            linewidth=2,
        )

    # Plot 4: Steered combined representations (SAME coordinate system)
    ax = axes[1, 1]

    # Create DataFrame for plotting
    df_combined_steer = pd.DataFrame(
        {
            "PC1": X_pca_combined_steer[:, 0],
            "PC2": X_pca_combined_steer[:, 1],
            "response": y_combined,
            "toxicity": labels_combined,
        }
    )

    sns.scatterplot(
        data=df_combined_steer,
        x="PC1",
        y="PC2",
        hue="toxicity",
        style="response",
        ax=ax,
        alpha=0.7,
        s=50,
    )
    ax.set_xlim(x_comb_min, x_comb_max)
    ax.set_ylim(y_comb_min, y_comb_max)
    ax.set_title(f"Steered - Combined Representations (α={steering_alpha})")
    ax.grid(True, alpha=0.3)

    # Same decision boundary
    if abs(w_pca_combined[1]) > 1e-8:
        ax.plot(
            x_vals_combined,
            y_vals_combined,
            "k--",
            label="Decision boundary",
            linewidth=2,
        )

    # Calculate and display movement statistics
    diff_movement = X_pca_diff_steer - X_pca_diff_orig
    combined_movement = X_pca_combined_steer - X_pca_combined_orig

    diff_movement_magnitude = np.mean(np.linalg.norm(diff_movement, axis=1))
    combined_movement_magnitude = np.mean(np.linalg.norm(combined_movement, axis=1))

    print("Boundary comparison movement stats:")
    print(f"  Difference vectors movement: {diff_movement_magnitude:.4f}")
    print(f"  Combined representations movement: {combined_movement_magnitude:.4f}")

    plt.suptitle(f"FIXED Boundary Comparison - Layer {best_layer}", fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"FIXED boundary comparison plot saved to: {save_path}")


# =============================================================================
# COMPREHENSIVE COMPARISON VISUALIZATION FUNCTIONS (RESTORED)
# =============================================================================


def create_comprehensive_comparison_visualizations(
    comparison_df, best_layer, steering_alpha, plots_dir
):
    """
    Create comprehensive comparison visualization plots.
    Restored: This function creates the missing comparison plots:
    - heatmap_comparison_layer_X_alpha_Y.png
    - line_comparison_layer_X_alpha_Y.png
    - steering_layer_bars_layer_X_alpha_Y.png
    - difference_analysis_layer_X_alpha_Y.png
    """
    print("Creating comprehensive comparison visualizations...")

    # 1. Heatmap comparison
    heatmap_path = (
        plots_dir / f"heatmap_comparison_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_metrics_heatmap(comparison_df, best_layer, steering_alpha, heatmap_path)

    # 2. Line comparison
    line_path = (
        plots_dir / f"line_comparison_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_metrics_line_plot(comparison_df, best_layer, steering_alpha, line_path)

    # 3. Steering layer bars
    bars_path = (
        plots_dir / f"steering_layer_bars_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_steering_bars_plot(comparison_df, best_layer, steering_alpha, bars_path)

    # 4. Difference analysis
    diff_path = (
        plots_dir / f"difference_analysis_layer_{best_layer}_alpha_{steering_alpha}.png"
    )
    _create_difference_analysis_plot(
        comparison_df, best_layer, steering_alpha, diff_path
    )

    return [heatmap_path, line_path, bars_path, diff_path]


def _create_metrics_heatmap(comparison_df, best_layer, steering_alpha, save_path):
    """Create heatmap visualization of metrics comparison."""
    # Extract base metrics (original vs steered)
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # Create data for heatmap
    heatmap_data = []
    for metric in base_metrics[:6]:  # Limit to first 6 metrics for visibility
        orig_values = comparison_df[f"{metric}_original"].values
        steered_values = comparison_df[f"{metric}_steered"].values
        heatmap_data.extend([orig_values, steered_values])

    heatmap_array = np.array(heatmap_data)

    # Create labels
    layer_labels = [f"Layer {i}" for i in comparison_df.index]
    metric_labels = []
    for metric in base_metrics[:6]:
        metric_labels.extend([f"{metric}_orig", f"{metric}_steer"])

    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        heatmap_array,
        xticklabels=layer_labels,
        yticklabels=metric_labels,
        annot=False,
        cmap="RdYlBu_r",
        center=0,
        cbar_kws={"label": "Metric Value"},
    )

    # Highlight steering layer
    plt.axvline(x=best_layer + 0.5, color="red", linewidth=3, alpha=0.7)
    plt.axvline(x=best_layer - 0.5, color="red", linewidth=3, alpha=0.7)

    plt.title(
        f"Metrics Heatmap Comparison (Steering Layer {best_layer}, α={steering_alpha})"
    )
    plt.xlabel("Layers")
    plt.ylabel("Metrics")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap comparison saved to: {save_path}")


def _create_metrics_line_plot(comparison_df, best_layer, steering_alpha, save_path):
    """Create line plot visualization of metrics comparison."""
    # Extract base metrics
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # Create subplots for first 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    layers = comparison_df.index.values

    for i, metric in enumerate(base_metrics[:4]):
        ax = axes[i]

        orig_values = comparison_df[f"{metric}_original"].values
        steered_values = comparison_df[f"{metric}_steered"].values

        ax.plot(layers, orig_values, "b-o", label="Original", linewidth=2, markersize=6)
        ax.plot(
            layers, steered_values, "r-s", label="Steered", linewidth=2, markersize=6
        )

        # Highlight steering layer
        ax.axvline(
            best_layer,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Steering Layer {best_layer}",
            alpha=0.7,
        )

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Metrics Line Comparison (α={steering_alpha})", fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Line comparison saved to: {save_path}")


def _create_steering_bars_plot(comparison_df, best_layer, steering_alpha, save_path):
    """Create bar plot showing steering effects."""
    # Extract percent changes for key metrics
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # Focus on steering layer
    steering_data = comparison_df.loc[best_layer]

    # Get percent changes for top metrics
    metrics_to_plot = base_metrics[:6]
    percent_changes = [
        steering_data[f"{metric}_percent_change"] for metric in metrics_to_plot
    ]

    # Create bar plot
    plt.figure(figsize=(12, 8))

    colors = ["red" if pc < 0 else "green" for pc in percent_changes]
    bars = plt.bar(
        range(len(metrics_to_plot)), percent_changes, color=colors, alpha=0.7
    )

    # Add value labels on bars
    for i, (bar, pc) in enumerate(zip(bars, percent_changes)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (1 if height > 0 else -3),
            f"{pc:.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
        )

    plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Metrics")
    plt.ylabel("Percent Change (%)")
    plt.title(f"Steering Effects on Layer {best_layer} (α={steering_alpha})")
    plt.xticks(
        range(len(metrics_to_plot)),
        [metric.replace("_", " ").title() for metric in metrics_to_plot],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Steering bars plot saved to: {save_path}")


def _create_difference_analysis_plot(
    comparison_df, best_layer, steering_alpha, save_path
):
    """Create difference analysis visualization."""
    # Extract base metrics
    base_metrics = []
    for col in comparison_df.columns:
        if col.endswith("_original"):
            base_metrics.append(col.replace("_original", ""))

    # Create subplots: absolute differences and percent changes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    layers = comparison_df.index.values

    # Plot 1: Absolute differences for key metrics
    for i, metric in enumerate(base_metrics[:3]):
        diff_values = comparison_df[f"{metric}_diff"].values
        ax1.plot(
            layers,
            diff_values,
            "o-",
            label=metric.replace("_", " ").title(),
            linewidth=2,
            markersize=6,
        )

    ax1.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Layer {best_layer}",
        alpha=0.7,
    )
    ax1.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Absolute Difference (Steered - Original)")
    ax1.set_title("Absolute Differences Across Layers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percent changes for key metrics
    for i, metric in enumerate(base_metrics[:3]):
        pct_values = comparison_df[f"{metric}_percent_change"].values
        ax2.plot(
            layers,
            pct_values,
            "s-",
            label=metric.replace("_", " ").title(),
            linewidth=2,
            markersize=6,
        )

    ax2.axvline(
        best_layer,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Steering Layer {best_layer}",
        alpha=0.7,
    )
    ax2.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Percent Change (%)")
    ax2.set_title("Percent Changes Across Layers")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Focusing on steering layer effects
    steering_metrics = base_metrics[:6]
    steering_diffs = [
        comparison_df.loc[best_layer, f"{metric}_diff"] for metric in steering_metrics
    ]

    colors = ["red" if diff < 0 else "green" for diff in steering_diffs]
    bars = ax3.bar(
        range(len(steering_metrics)), steering_diffs, color=colors, alpha=0.7
    )

    ax3.axhline(0, color="black", linestyle="-", linewidth=1)
    ax3.set_xlabel("Metrics")
    ax3.set_ylabel("Absolute Difference")
    ax3.set_title(f"Steering Layer {best_layer} - Absolute Changes")
    ax3.set_xticks(range(len(steering_metrics)))
    ax3.set_xticklabels(
        [metric.replace("_", " ").title() for metric in steering_metrics],
        rotation=45,
        ha="right",
    )
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Distribution of changes
    all_pct_changes = []
    for metric in base_metrics[:4]:
        pct_values = comparison_df[f"{metric}_percent_change"].values
        all_pct_changes.extend(pct_values)

    ax4.hist(all_pct_changes, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax4.axvline(0, color="red", linestyle="--", linewidth=2, label="No Change")
    ax4.set_xlabel("Percent Change (%)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of All Percent Changes")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Difference Analysis (α={steering_alpha})", fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Difference analysis plot saved to: {save_path}")


if __name__ == "__main__":
    print("This is the FIXED version of steering analysis functions.")
    print(
        "Key fix: All PCA transformations use the SAME coordinate system for original and steered data."
    )
    print("This allows you to see how steering actually moved the data points.")
