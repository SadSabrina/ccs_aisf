import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set the style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")  # Use a more vibrant color palette

# Custom color palette for better visualization
CUSTOM_PALETTE = {
    "hate": "#FF6B6B",  # Coral red
    "safe": "#4ECDC4",  # Turquoise
    "steering": "#45B7D1",  # Sky blue
    "background": "#F7F7F7",  # Light gray
    "grid": "#E0E0E0",  # Lighter gray for grid
}


def safe_pca_transform(data, n_components=2):
    """Perform PCA with numerical stability checks"""
    # Convert to numpy if tensor
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    # Check for NaN or Inf values and replace them
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

    # Center the data with numerical stability
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Scale the data with numerical stability
    std = np.std(centered, axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    scaled = centered / std

    # Clip values to prevent overflow
    scaled = np.clip(scaled, -1e6, 1e6)

    # Check for NaN or Inf values again after scaling
    if np.any(np.isnan(scaled)) or np.any(np.isinf(scaled)):
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1e6, neginf=-1e6)

    # Apply PCA with full SVD solver for better numerical stability
    pca = PCA(n_components=n_components, svd_solver="full")

    # Add small epsilon to diagonal for numerical stability
    scaled = scaled + np.eye(scaled.shape[0], scaled.shape[1]) * 1e-10

    transformed = pca.fit_transform(scaled)

    # Final check for NaN or Inf values
    if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=1e6, neginf=-1e6)

    return transformed


def plot_pca_or_tsne_layerwise(
    X_pos,
    X_neg,
    hue,
    standardize=True,
    reshape=None,
    n_components=5,
    components=None,
    mode="pca",
    plot_title=None,
    save_path=None,
):
    """
    PCA or T-SNE-clustering for each hidden layer plot

    Parameters:
        X_pos (np.ndarray): Positive (yes) samples, shape (n_samples, n_layers, hidden_dim)
        X_neg (np.ndarray): Negative (no) samples, shape (n_samples, n_layers, hidden_dim)
        hue (np.ndarray): Labels for coloring points
        standardize (bool): If standardization is needed before PCA
        reshape (list): If data needs reshaping
        n_components (int): Number of PCA/TSNE components
        components (list): 2 components to plot
        mode (str): 'pca' or 'tsne'
        plot_title (str): Figure suptitle
        save_path (str): Path to save the plot
    """
    if components is None:
        components = [0, 1]

    if len(X_pos.shape) == 2:
        if reshape is None:
            raise ValueError("reshape parameter required when input is 2D")
        X_pos = X_pos.reshape(reshape[0], reshape[1], -1)
        X_neg = X_neg.reshape(reshape[0], reshape[1], -1)

    n_layers = X_pos.shape[1]

    try:
        fig, axes = plt.subplots((n_layers - 1) // 6 + 1, 6, figsize=(24, 13))
    except:
        fig, axes = plt.subplots((n_layers - 1) // 6 + 1 + 1, 6, figsize=(24, 13))

    axes = axes.flatten()

    for layer_idx in range(1, n_layers):
        ax = axes[layer_idx - 1]

        X_pos_layer = X_pos[:, layer_idx, :]
        X_neg_layer = X_neg[:, layer_idx, :]

        # Combining hidden states
        states_data = X_pos_layer - X_neg_layer

        # Standardization
        if standardize:
            states_data = (states_data - states_data.mean(axis=0)) / (
                states_data.std(axis=0) + 1e-10
            )

        if mode == "pca":
            projector = PCA(n_components=n_components)
        elif mode == "tsne":
            projector = TSNE(n_components=n_components, metric="cosine")
        else:
            raise ValueError("mode must be 'pca' or 'tsne'")

        X_proj = projector.fit_transform(states_data)

        # Plot
        sns.scatterplot(
            data=pd.DataFrame(X_proj),
            x=X_proj[:, components[0]],
            y=X_proj[:, components[1]],
            hue=hue,
            ax=ax,
        )
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.legend().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=12)

    if plot_title:
        fig.suptitle(plot_title, fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_all_steering_vectors(
    results,
    hate_vectors=None,
    safe_vectors=None,
    log_base=None,
    all_steering_vectors=None,
    hate_types=None,
    safe_types=None,
):
    """
    Plot all steering vectors across layers.

    Args:
        results: List of dicts, each with keys 'hate_mean_vector', 'safe_mean_vector', 'steering_vector'
        hate_vectors: List of arrays, each with shape (n_samples, 1, 768) or (n_samples, 768)
        safe_vectors: List of arrays, each with shape (n_samples, 1, 768) or (n_samples, 768)
        log_base: Base path for saving the plot
        all_steering_vectors: Dict of different steering vectors with their properties
        hate_types: List of types for hate vectors
        safe_types: List of types for safe vectors
    """
    n_layers = len(results)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 6 * n_layers))

    # Handle single layer case
    if n_layers == 1:
        axes = [axes]

    # Define colors and markers for each category
    category_styles = {
        "hate_yes": {
            "color": "#FF0000",  # Bright red
            "marker": "o",
            "label": "Hate (Yes)",
        },
        "hate_no": {
            "color": "#0066CC",  # Medium blue
            "marker": "s",
            "label": "Hate (No)",
        },
        "safe_yes": {
            "color": "#3399FF",  # Light blue
            "marker": "^",
            "label": "Safe (Yes)",
        },
        "safe_no": {
            "color": "#CC0000",  # Dark red
            "marker": "v",
            "label": "Safe (No)",
        },
    }

    for layer_idx in range(n_layers):
        layer_data = results[layer_idx]
        ax = axes[layer_idx]

        # Get vectors and ensure correct shapes
        hate_mean_vector = layer_data["hate_mean_vector"]
        safe_mean_vector = layer_data["safe_mean_vector"]
        steering_vector = layer_data["steering_vector"]

        # Print shapes for debugging
        print(
            f"Layer {layer_idx} vector shapes - hate_mean: {hate_mean_vector.shape}, "
            f"safe_mean: {safe_mean_vector.shape}, steering: {steering_vector.shape}"
        )

        # Plot all data points if provided
        if hate_vectors is not None and safe_vectors is not None:
            # Get layer-specific vectors
            layer_hate_vectors = hate_vectors[layer_idx]
            layer_safe_vectors = safe_vectors[layer_idx]

            print(
                f"Layer {layer_idx} data shapes - hate: {layer_hate_vectors.shape}, safe: {layer_safe_vectors.shape}"
            )

            # Reshape vectors if they're 3D
            if len(layer_hate_vectors.shape) == 3:
                hate_vectors_2d = layer_hate_vectors.reshape(
                    layer_hate_vectors.shape[0], -1
                )
                safe_vectors_2d = layer_safe_vectors.reshape(
                    layer_safe_vectors.shape[0], -1
                )
                print(
                    f"Reshaped to 2D - hate: {hate_vectors_2d.shape}, safe: {safe_vectors_2d.shape}"
                )
            else:
                hate_vectors_2d = layer_hate_vectors
                safe_vectors_2d = layer_safe_vectors

            # Stack all vectors for PCA
            all_vectors = np.vstack([hate_vectors_2d, safe_vectors_2d])

            # Check for numerical issues and add jitter if needed
            if np.allclose(hate_vectors_2d, safe_vectors_2d, atol=1e-8):
                print(
                    "Warning: Hate and safe vectors are nearly identical, adding random jitter"
                )
                # Add small random noise to avoid numerical instability
                noise_scale = 1e-6  # Very small scale
                all_vectors += np.random.normal(0, noise_scale, all_vectors.shape)

            # Add a tiny bit of regularization to avoid numerical issues
            epsilon = 1e-8
            all_vectors_reg = all_vectors + np.random.normal(
                0, epsilon, all_vectors.shape
            )

            # Normalize vectors to avoid numerical issues in PCA
            all_vectors_mean = np.mean(all_vectors_reg, axis=0, keepdims=True)
            all_vectors_std = (
                np.std(all_vectors_reg, axis=0, keepdims=True) + 1e-10
            )  # Add small value to avoid divide by zero
            all_vectors_norm = (all_vectors_reg - all_vectors_mean) / all_vectors_std

            # Stack mean vectors and steering vectors for consistent projection
            mean_vectors_list = [
                hate_mean_vector.reshape(1, -1),
                safe_mean_vector.reshape(1, -1),
            ]

            # Add all steering vectors
            steering_vectors_list = []
            if all_steering_vectors is not None:
                for name, data in all_steering_vectors.items():
                    steering_vectors_list.append(data["vector"].reshape(1, -1))
            else:
                # Add the default steering vector if no multiple vectors provided
                steering_vectors_list.append(steering_vector.reshape(1, -1))

            # Stack all vectors for PCA projection
            all_mean_steering_vectors = np.vstack(
                mean_vectors_list + steering_vectors_list
            )

            # Add the same regularization to mean vectors for consistency
            all_mean_steering_vectors_reg = (
                all_mean_steering_vectors - all_vectors_mean
            ) / all_vectors_std

            try:
                # Fit PCA to all data points to get a good projection space
                pca = PCA(n_components=2, svd_solver="full")
                all_2d = pca.fit_transform(all_vectors_norm)

                # Transform mean vectors using the same PCA
                mean_2d = pca.transform(all_mean_steering_vectors_reg)
            except (ValueError, RuntimeWarning) as e:
                print(
                    f"Warning: PCA failed with error: {e}. Using simple dimension selection."
                )
                # Fallback: handle arrays with small dimensions correctly
                if all_vectors_norm.shape[1] <= 2:
                    # If input has only 1 or 2 dimensions, use them directly
                    if all_vectors_norm.shape[1] == 1:
                        # For 1D data, create a synthetic second dimension with small random values
                        all_2d = np.column_stack(
                            [
                                all_vectors_norm[:, 0],
                                np.random.normal(
                                    0, 0.01, size=all_vectors_norm.shape[0]
                                ),
                            ]
                        )
                        mean_2d = np.column_stack(
                            [
                                all_mean_steering_vectors_reg[:, 0],
                                np.random.normal(
                                    0, 0.01, size=all_mean_steering_vectors_reg.shape[0]
                                ),
                            ]
                        )
                    else:
                        # For 2D data, use both dimensions
                        all_2d = all_vectors_norm
                        mean_2d = all_mean_steering_vectors_reg
                else:
                    # For higher dimensional data, select two dimensions with highest variance
                    var = np.var(all_vectors_norm, axis=0)
                    idx = np.argsort(var)[-2:]
                    all_2d = all_vectors_norm[:, idx]
                    mean_2d = all_mean_steering_vectors_reg[:, idx]

            # Split back into hate and safe
            n_hate = len(hate_vectors_2d)
            hate_2d = all_2d[:n_hate]
            safe_2d = all_2d[n_hate:]

            # Get transformed mean vectors
            hate_mean_2d = mean_2d[0]
            safe_mean_2d = mean_2d[1]

            # Get transformed steering vectors
            steering_2d_list = mean_2d[2:]  # All steering vectors start from index 2

            # If we have type information, use it to color the points differently
            if hate_types is not None and safe_types is not None:
                # Calculate indices for each category first
                hate_yes_indices = [
                    j for j, t in enumerate(hate_types) if t == "hate_yes"
                ]
                hate_no_indices = [
                    j for j, t in enumerate(hate_types) if t == "hate_no"
                ]
                safe_yes_indices = [
                    j for j, t in enumerate(safe_types) if t == "safe_yes"
                ]
                safe_no_indices = [
                    j for j, t in enumerate(safe_types) if t == "safe_no"
                ]

                # Count how many specialized means we have
                n_specialized_means = 0
                if "hate_yes_indices" in locals() and len(hate_yes_indices) > 0:
                    n_specialized_means += 1
                if "hate_no_indices" in locals() and len(hate_no_indices) > 0:
                    n_specialized_means += 1
                if "safe_yes_indices" in locals() and len(safe_yes_indices) > 0:
                    n_specialized_means += 1
                if "safe_no_indices" in locals() and len(safe_no_indices) > 0:
                    n_specialized_means += 1

                # For each type, plot the corresponding points
                for category, style in category_styles.items():
                    if category.startswith("hate_"):
                        # Find indices of this category in the hate data
                        if category == "hate_yes":
                            indices = hate_yes_indices
                        else:  # hate_no
                            indices = hate_no_indices

                        if indices:  # Only plot if we have points of this type
                            ax.scatter(
                                hate_2d[indices, 0],
                                hate_2d[indices, 1],
                                color=style["color"],
                                alpha=0.5,
                                marker=style["marker"],
                                s=20,
                                edgecolors="black",
                                linewidths=0.5,
                                label=style["label"],
                                zorder=5,
                            )
                    else:  # safe category
                        # Find indices of this category in the safe data
                        if category == "safe_yes":
                            indices = safe_yes_indices
                        else:  # safe_no
                            indices = safe_no_indices

                        if indices:  # Only plot if we have points of this type
                            ax.scatter(
                                safe_2d[indices, 0],
                                safe_2d[indices, 1],
                                color=style["color"],
                                alpha=0.5,
                                marker=style["marker"],
                                s=20,
                                edgecolors="black",
                                linewidths=0.5,
                                label=style["label"],
                                zorder=5,
                            )
            else:
                # Fall back to original plotting if type information is not available
                ax.scatter(
                    hate_2d[:, 0],
                    hate_2d[:, 1],
                    color="#FF0000",  # Red for hate
                    alpha=0.5,
                    marker="o",
                    s=20,
                    edgecolors="black",
                    linewidths=0.5,
                    label="Hate Data",
                    zorder=5,
                )
                ax.scatter(
                    safe_2d[:, 0],
                    safe_2d[:, 1],
                    color="#0000FF",  # Blue for safe
                    alpha=0.5,
                    marker="^",
                    s=20,
                    edgecolors="black",
                    linewidths=0.5,
                    label="Safe Data",
                    zorder=6,
                )

            # Plot mean vectors with bright colors and smaller arrows
            ax.quiver(
                0,
                0,
                hate_mean_2d[0],
                hate_mean_2d[1],
                color="#FF0000",  # Bright red
                label="Hate Mean",
                scale_units="xy",
                scale=4,  # Higher scale makes arrows smaller
                width=0.008,  # Thinner arrow
                headwidth=5,  # Smaller head width
                headlength=6,  # Smaller head length
                alpha=1.0,
                zorder=10,  # Make sure it's on top
            )
            ax.quiver(
                0,
                0,
                safe_mean_2d[0],
                safe_mean_2d[1],
                color="#0000FF",  # Bright blue
                label="Safe Mean",
                scale_units="xy",
                scale=4,  # Higher scale makes arrows smaller
                width=0.008,  # Thinner arrow
                headwidth=5,  # Smaller head width
                headlength=6,  # Smaller head length
                alpha=1.0,
                zorder=10,  # Make sure it's on top
            )

            # Plot steering vectors
            steering_idx = 0
            if all_steering_vectors is not None:
                for name, data in all_steering_vectors.items():
                    if steering_idx < len(mean_2d):
                        ax.quiver(
                            0,
                            0,
                            mean_2d[steering_idx][0],
                            mean_2d[steering_idx][1],
                            color=data["color"],
                            label=data["label"],
                            scale_units="xy",
                            scale=4,
                            width=0.008,
                            headwidth=5,
                            headlength=6,
                            alpha=1.0,
                            zorder=11,
                        )
                        steering_idx += 1
            elif steering_vectors_list:
                # Plot default steering vector if specific ones aren't provided
                if steering_idx < len(mean_2d):
                    ax.quiver(
                        0,
                        0,
                        mean_2d[steering_idx][0],
                        mean_2d[steering_idx][1],
                        color="#00FF00",  # Green for steering vector
                        label="Steering Vector",
                        scale_units="xy",
                        scale=4,
                        width=0.008,
                        headwidth=5,
                        headlength=6,
                        alpha=1.0,
                        zorder=11,
                    )

            # Set axis limits based on all data
            x_min, x_max = np.min(all_2d[:, 0]), np.max(all_2d[:, 0])
            y_min, y_max = np.min(all_2d[:, 1]), np.max(all_2d[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            # If no data points provided, just plot the mean vectors using their own PCA
            mean_vectors_list = [
                hate_mean_vector.reshape(1, -1),
                safe_mean_vector.reshape(1, -1),
            ]

            # Add all steering vectors
            steering_vectors_list = []
            if all_steering_vectors is not None:
                for name, data in all_steering_vectors.items():
                    steering_vectors_list.append(data["vector"].reshape(1, -1))
            else:
                # Add the default steering vector if no multiple vectors provided
                steering_vectors_list.append(steering_vector.reshape(1, -1))

            # Stack all vectors for PCA projection
            all_mean_steering_vectors = np.vstack(
                mean_vectors_list + steering_vectors_list
            )

            pca = PCA(n_components=2)
            mean_2d = pca.fit_transform(all_mean_steering_vectors)

            # Get transformed vectors
            hate_mean_2d = mean_2d[0]
            safe_mean_2d = mean_2d[1]
            steering_2d_list = mean_2d[2:]  # All steering vectors start from index 2

            # Plot mean vectors with smaller arrows
            ax.quiver(
                0,
                0,
                hate_mean_2d[0],
                hate_mean_2d[1],
                color="#FF0000",  # Bright red
                label="Hate Mean",
                scale=4,  # Higher scale makes arrows smaller
                width=0.008,  # Thinner arrow
                headwidth=4,  # Smaller head width
                headlength=5,  # Smaller head length
                alpha=1.0,
            )
            ax.quiver(
                0,
                0,
                safe_mean_2d[0],
                safe_mean_2d[1],
                color="#0000FF",  # Bright blue
                label="Safe Mean",
                scale=4,  # Higher scale makes arrows smaller
                width=0.008,  # Thinner arrow
                headwidth=4,  # Smaller head width
                headlength=5,  # Smaller head length
                alpha=1.0,
            )

            # Plot steering vectors
            steering_idx = 0
            if all_steering_vectors is not None:
                for name, data in all_steering_vectors.items():
                    if steering_idx < len(mean_2d):
                        ax.quiver(
                            0,
                            0,
                            mean_2d[steering_idx][0],
                            mean_2d[steering_idx][1],
                            color=data["color"],
                            label=data["label"],
                            scale_units="xy",
                            scale=4,
                            width=0.008,
                            headwidth=5,
                            headlength=6,
                            alpha=1.0,
                            zorder=11,
                        )
                        steering_idx += 1
            elif steering_vectors_list:
                # Plot default steering vector if specific ones aren't provided
                if steering_idx < len(mean_2d):
                    ax.quiver(
                        0,
                        0,
                        mean_2d[steering_idx][0],
                        mean_2d[steering_idx][1],
                        color="#00FF00",  # Green for steering vector
                        label="Steering Vector",
                        scale_units="xy",
                        scale=4,
                        width=0.008,
                        headwidth=5,
                        headlength=6,
                        alpha=1.0,
                        zorder=11,
                    )

            # Set axis limits
            max_val = max(
                np.max(np.abs(mean_2d[:, 0])),
                np.max(np.abs(mean_2d[:, 1])),
            )
            ax.set_xlim(-max_val * 1.2, max_val * 1.2)
            ax.set_ylim(-max_val * 1.2, max_val * 1.2)

        # Add layer info to the title
        ax.set_title(f"Layer {layer_idx} Steering Vectors")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")

    plt.tight_layout()

    if log_base is not None:
        save_path = os.path.join(log_base, "all_steering_vectors.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved steering vectors plot to {save_path}")

    plt.close()  # Close the plot to avoid memory issues
    return plt


def plot_performance_across_layers(results, metric="accuracy", save_path=None):
    """Plot performance metrics across layers.

    Args:
        results: List of dictionaries containing layer results
        metric: Metric to plot ('accuracy', 'auc', or 'silhouette')
        save_path: Path to save the plot
    """
    layers = [r["layer_idx"] for r in results]

    # Get baseline metrics
    baseline_metrics = [r["final_metrics"]["base_metrics"][metric] for r in results]

    # Get metrics for each steering coefficient
    steering_coefs = [0.0, 0.5, 1.0, 2.0, 5.0]
    coef_metrics = {coef: [] for coef in steering_coefs}

    for r in results:
        for coef in steering_coefs:
            coef_str = f"coef_{coef}"
            if coef_str in r and metric in r[coef_str]:
                coef_metrics[coef].append(r[coef_str][metric])
            else:
                coef_metrics[coef].append(None)

    plt.figure(figsize=(12, 6))

    # Plot baseline
    plt.plot(layers, baseline_metrics, "k--", label="Baseline")

    # Plot steering coefficients
    for coef in steering_coefs:
        plt.plot(layers, coef_metrics[coef], label=f"Coef={coef}")

    plt.xlabel("Layer")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} Across Layers")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_all_layer_vectors(results, save_dir):
    """Plot all layer vectors in a grid."""
    n_layers = len(results)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for i, layer_data in enumerate(results):
        # Extract vectors and ensure correct shapes
        hate_mean = layer_data["hate_mean_vector"]
        safe_mean = layer_data["safe_mean_vector"]
        steering = layer_data["steering_vector"]

        # Print shapes for debugging
        print(
            f"Layer {i} shapes - hate_mean: {hate_mean.shape}, safe_mean: {safe_mean.shape}, steering: {steering.shape}"
        )

        # Ensure vectors are properly shaped for PCA
        hate_mean_vec = hate_mean.reshape(1, -1)
        safe_mean_vec = safe_mean.reshape(1, -1)
        steering_vec = steering.reshape(1, -1)

        # Add small jitter to avoid numerical issues
        epsilon = 1e-8
        hate_mean_vec += np.random.normal(0, epsilon, hate_mean_vec.shape)
        safe_mean_vec += np.random.normal(0, epsilon, safe_mean_vec.shape)
        steering_vec += np.random.normal(0, epsilon, steering_vec.shape)

        # Project to 2D using PCA with safeguards
        vectors = np.vstack([hate_mean_vec, safe_mean_vec, steering_vec])

        try:
            # Standardize before PCA
            vectors_mean = np.mean(vectors, axis=0, keepdims=True)
            vectors_std = (
                np.std(vectors, axis=0, keepdims=True) + 1e-10
            )  # Avoid divide by zero
            vectors_norm = (vectors - vectors_mean) / vectors_std

            pca = PCA(n_components=2, svd_solver="full")
            vectors_2d = pca.fit_transform(vectors_norm)
        except (ValueError, RuntimeWarning) as e:
            print(
                f"Warning in layer {i}: PCA failed with error: {e}. Using simple dimension selection."
            )
            # Fallback: just select 2 dimensions with highest variance
            var = np.var(vectors, axis=0)
            idx = np.argsort(var)[-2:]
            vectors_2d = vectors[:, idx]

        ax = axes[i]
        ax.quiver(
            0,
            0,
            vectors_2d[0, 0],
            vectors_2d[0, 1],
            color="#FF0000",
            label="Hate Mean",
            scale=1,
            width=0.015,
            headwidth=5,
            headlength=7,
        )
        ax.quiver(
            0,
            0,
            vectors_2d[1, 0],
            vectors_2d[1, 1],
            color="#0000FF",
            label="Safe Mean",
            scale=1,
            width=0.015,
            headwidth=5,
            headlength=7,
        )
        ax.quiver(
            0,
            0,
            vectors_2d[2, 0],
            vectors_2d[2, 1],
            color="#00FF00",
            label="Steering Vector",
            scale=1,
            width=0.015,
            headwidth=5,
            headlength=7,
        )

        ax.set_title(f"Layer {i}")
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.grid(True)
        if i == 0:
            ax.legend()

    # Hide unused subplots
    for j in range(n_layers, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "all_layer_vectors.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    return fig


def visualize_decision_boundary(
    ccs, hate_vectors, safe_vectors, steering_vector, log_base=None
):
    """Visualize the decision boundary of the CCS probe in the steering vector direction."""
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize steering vector
    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm > 1e-10:
        steering_vector = steering_vector / steering_norm

    # Combine data
    X_combined = np.vstack([hate_vectors, safe_vectors])
    labels = np.concatenate([np.zeros(len(hate_vectors)), np.ones(len(safe_vectors))])

    # Project data to 2D for visualization
    # First component: steering vector direction
    # Second component: PCA of residuals
    projections = []

    # Project onto steering vector
    projection1 = np.array([np.dot(x, steering_vector) for x in X_combined])
    projections.append(projection1)

    # Compute residuals
    residuals = X_combined - np.outer(projection1, steering_vector)

    # Find second direction (orthogonal to steering vector)
    pca = PCA(n_components=1)
    pca.fit(residuals)
    second_direction = pca.components_[0]

    # Project onto second direction
    projection2 = np.array([np.dot(x, second_direction) for x in X_combined])
    projections.append(projection2)

    # Create 2D projections
    X_2d = np.column_stack(projections)

    # Create grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Reconstruct grid points in original space
    grid_points = np.array([xx.ravel(), yy.ravel()]).T
    original_space = np.outer(grid_points[:, 0], steering_vector) + np.outer(
        grid_points[:, 1], second_direction
    )

    # Predict on grid points
    X_placeholder = np.zeros_like(original_space)
    grid_preds, _ = ccs.predict(original_space, X_placeholder)
    grid_preds = grid_preds.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, grid_preds, alpha=0.3, cmap="RdBu")

    # Plot data points
    colors = np.array(["#D7263D", "#1B98E0"])  # Red for hate, blue for safe
    edge_color = "k"  # Black edge for contrast

    for label, color in zip([0, 1], colors):
        idx = labels == label
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            c=color,
            edgecolor=edge_color,
            s=70,
            alpha=0.85,
            label="Hate" if label == 0 else "Safe",
        )

    # Add steering vector direction
    ax.arrow(
        0,
        0,
        1,
        0,
        color="black",
        width=0.01,
        head_width=0.1,
        head_length=0.1,
        length_includes_head=True,
        label="Steering Direction",
    )

    ax.set_xlabel("Steering Vector Direction")
    ax.set_ylabel("Orthogonal Direction")
    ax.set_title("CCS Probe Decision Boundary in Steering Space")
    ax.legend(loc="upper right", fontsize=12)

    # Add descriptive text
    text = """
    Description: This plot shows the decision boundary of the CCS probe in the space defined by the steering vector and its orthogonal complement.
    
    Ideal Case:
    - Clear separation between hate (red) and safe (blue) content
    - Decision boundary should be roughly perpendicular to the steering direction
    - Points should cluster into two distinct groups
    
    Interpretation:
    - The steering vector direction (horizontal axis) shows how content changes when steered
    - The orthogonal direction (vertical axis) shows variations that preserve the steering effect
    - The decision boundary (colored regions) shows where the probe switches between hate and safe predictions
    - A clear boundary indicates the probe can reliably distinguish between content types
    """

    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        wrap=True,
    )

    # Adjust layout to make room for text
    plt.tight_layout(rect=(0, 0.15, 1, 1))

    if log_base:
        plt.savefig(f"{log_base}_decision_boundary.png", dpi=300, bbox_inches="tight")

    return fig


def plot_all_decision_boundaries(layers_data, log_base=None):
    """Plot decision boundaries for all layers as subplots in a single figure."""
    n_layers = len(layers_data)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for i, layer in enumerate(layers_data):
        ccs = layer["ccs"]
        hate_vectors = layer["hate_vectors"]
        safe_vectors = layer["safe_vectors"]
        steering_vector = layer["steering_vector"]
        layer_idx = layer.get("layer_idx", i)
        ax = axes[i]

        # Normalize steering vector
        steering_norm = np.linalg.norm(steering_vector)
        if steering_norm > 1e-10:
            steering_vector = steering_vector / steering_norm

        # Combine data
        X_combined = np.vstack([hate_vectors, safe_vectors])
        labels = np.concatenate(
            [np.zeros(len(hate_vectors)), np.ones(len(safe_vectors))]
        )

        # Project data to 2D for visualization
        projection1 = np.array([np.dot(x, steering_vector) for x in X_combined])
        residuals = X_combined - np.outer(projection1, steering_vector)
        pca = PCA(n_components=1)
        pca.fit(residuals)
        second_direction = pca.components_[0]
        projection2 = np.array([np.dot(x, second_direction) for x in X_combined])
        X_2d = np.column_stack([projection1, projection2])

        # Create grid for decision boundary
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid_points = np.array([xx.ravel(), yy.ravel()]).T
        original_space = np.outer(grid_points[:, 0], steering_vector) + np.outer(
            grid_points[:, 1], second_direction
        )
        X_placeholder = np.zeros_like(original_space)
        grid_preds, _ = ccs.predict(original_space, X_placeholder)
        grid_preds = grid_preds.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, grid_preds, alpha=0.3, cmap="RdBu")
        colors = np.array(["#D7263D", "#1B98E0"])  # Red for hate, blue for safe
        edge_color = "k"  # Black edge for contrast

        for label, color in zip([0, 1], colors):
            idx = labels == label
            ax.scatter(
                X_2d[idx, 0],
                X_2d[idx, 1],
                c=color,
                edgecolor=edge_color,
                s=70,
                alpha=0.85,
                label="Hate" if label == 0 else "Safe",
            )
        ax.arrow(
            0,
            0,
            1,
            0,
            color="black",
            width=0.01,
            head_width=0.1,
            head_length=0.1,
            length_includes_head=True,
            label="Steering Direction",
        )
        ax.set_xlabel("Steering Vector Direction")
        ax.set_ylabel("Orthogonal Direction")
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right", fontsize=12)

    # Hide unused subplots
    for j in range(n_layers, len(axes)):
        axes[j].axis("off")

    # Add detailed description block
    description = (
        "This figure shows the decision boundaries of the CCS probe for each layer in the space defined by the steering vector (horizontal axis) and its orthogonal complement (vertical axis).\n\n"
        "**How to interpret:**\n"
        "- Each subplot corresponds to a different layer.\n"
        "- The colored regions show the model's predicted class (hate or safe) in the 2D projection.\n"
        "- The black arrow shows the direction of the steering vector.\n"
        "- Points are colored by their true class.\n"
        "- A clear, vertical decision boundary (perpendicular to the steering vector) indicates the probe can reliably distinguish between content types along the steering direction.\n\n"
        "**Ideal case:**\n"
        "- Hate and safe points form two distinct clusters separated by a sharp boundary.\n"
        "- The boundary is perpendicular to the steering direction.\n"
        "- The probe's predictions match the true classes.\n\n"
        "**Non-ideal case:**\n"
        "- Overlapping clusters or a fuzzy boundary indicate the probe struggles to distinguish between classes.\n"
        "- A boundary not aligned with the steering direction suggests the steering vector is not the most discriminative direction.\n"
        "\n"
        "**Axes:**\n"
        "- Horizontal: projection onto the steering vector (how much a point moves when steered).\n"
        "- Vertical: projection onto the main orthogonal direction (other variations).\n"
    )
    fig.text(
        0.5,
        0.01,
        description,
        ha="center",
        va="bottom",
        wrap=True,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    if log_base:
        plt.savefig(
            f"{log_base}_all_decision_boundaries.png", dpi=300, bbox_inches="tight"
        )
    return fig


def plot_vectors_all_strategies(
    layer_idx,
    all_strategy_data,
    current_strategy="last-token",
    save_path=None,
    all_steering_vectors=None,
):
    """
    Plot vectors for all three embedding strategies (mean, last-token, first-token) in a single figure.

    Args:
        layer_idx: Layer index for title
        all_strategy_data: Dictionary with data for all strategies
        current_strategy: The current strategy used in the model ("mean", "last-token", or "first-token")
        save_path: Path to save the plot
        all_steering_vectors: Dict of different steering vectors with their properties
    """
    # Create figure with 1 row and 3 columns for the three strategies
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Define the three strategies and titles
    strategies = ["mean", "last-token", "first-token"]
    strategy_titles = [
        "Mean Embedding",
        "Last Token Embedding",
        "First Token Embedding",
    ]

    # Define consistent colors and markers for each category
    category_styles = {
        "hate_yes": {
            "color": "#FF0000",  # Red
            "marker": "o",  # Round
            "label": "Hate Yes",
        },
        "hate_no": {
            "color": "#0000FF",  # Blue
            "marker": "o",  # Round
            "label": "Hate No",
        },
        "safe_yes": {
            "color": "#0000FF",  # Blue
            "marker": "^",  # Triangle up
            "label": "Safe Yes",
        },
        "safe_no": {
            "color": "#FF0000",  # Red
            "marker": "^",  # Triangle up
            "label": "Safe No",
        },
    }

    # Plot each strategy in its own subplot
    for i, (strategy, title) in enumerate(zip(strategies, strategy_titles)):
        ax = axes[i]

        # Get data for this strategy
        data = all_strategy_data[strategy]

        # Access individual category data
        hate_yes_vectors = data.get("hate_yes", None)
        hate_no_vectors = data.get("hate_no", None)
        safe_yes_vectors = data.get("safe_yes", None)
        safe_no_vectors = data.get("safe_no", None)

        # Make sure we have all necessary vectors
        if (
            hate_yes_vectors is None
            or hate_no_vectors is None
            or safe_yes_vectors is None
            or safe_no_vectors is None
        ):
            print(
                f"Warning: Missing some vector types for {strategy}. Using available data."
            )
            # Fall back to combined hate/safe if needed
            hate_vectors = data.get("hate", None)
            safe_vectors = data.get("safe", None)

            if hate_vectors is not None and safe_vectors is not None:
                # For standard datasets, we'll split each category (hate/safe) into two
                # equal parts to simulate hate_yes/hate_no and safe_yes/safe_no
                half_hate = len(hate_vectors) // 2
                half_safe = len(safe_vectors) // 2

                # Split each category
                hate_yes_vectors = hate_vectors[:half_hate]
                hate_no_vectors = hate_vectors[half_hate:]
                safe_yes_vectors = safe_vectors[:half_safe]
                safe_no_vectors = safe_vectors[half_safe:]

        # Reshape vectors if they're 3D
        for vector_name in [
            "hate_yes_vectors",
            "hate_no_vectors",
            "safe_yes_vectors",
            "safe_no_vectors",
        ]:
            vector = locals()[vector_name]
            if vector is not None and len(vector.shape) == 3:
                locals()[vector_name] = vector.reshape(vector.shape[0], -1)

        # Apply reshaping back to the variables
        hate_yes_vectors = locals()["hate_yes_vectors"]
        hate_no_vectors = locals()["hate_no_vectors"]
        safe_yes_vectors = locals()["safe_yes_vectors"]
        safe_no_vectors = locals()["safe_no_vectors"]

        # Stack all vectors for PCA
        all_vectors = np.vstack(
            [
                hate_yes_vectors if hate_yes_vectors is not None else np.array([]),
                hate_no_vectors if hate_no_vectors is not None else np.array([]),
                safe_yes_vectors if safe_yes_vectors is not None else np.array([]),
                safe_no_vectors if safe_no_vectors is not None else np.array([]),
            ]
        )

        # Check for empty data
        if all_vectors.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data available for this strategy",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            continue

        # Add tiny jitter to avoid numerical issues
        epsilon = 1e-8
        all_vectors_reg = all_vectors + np.random.normal(0, epsilon, all_vectors.shape)

        # Normalize vectors to avoid numerical issues in PCA
        all_vectors_mean = np.mean(all_vectors_reg, axis=0, keepdims=True)
        all_vectors_std = np.std(all_vectors_reg, axis=0, keepdims=True) + 1e-10
        all_vectors_norm = (all_vectors_reg - all_vectors_mean) / all_vectors_std

        # Prepare mean vectors
        mean_vectors = []

        # Calculate mean vectors for each category
        hate_yes_mean = (
            np.mean(hate_yes_vectors, axis=0).reshape(1, -1)
            if hate_yes_vectors.size > 0
            else None
        )
        hate_no_mean = (
            np.mean(hate_no_vectors, axis=0).reshape(1, -1)
            if hate_no_vectors.size > 0
            else None
        )
        safe_yes_mean = (
            np.mean(safe_yes_vectors, axis=0).reshape(1, -1)
            if safe_yes_vectors.size > 0
            else None
        )
        safe_no_mean = (
            np.mean(safe_no_vectors, axis=0).reshape(1, -1)
            if safe_no_vectors.size > 0
            else None
        )

        # Add available means to the list
        for mean_vector in [hate_yes_mean, hate_no_mean, safe_yes_mean, safe_no_mean]:
            if mean_vector is not None:
                mean_vectors.append(mean_vector)

        # Add steering vectors
        steering_vectors_list = []
        if all_steering_vectors is not None:
            for name, data in all_steering_vectors.items():
                steering_vectors_list.append(data["vector"].reshape(1, -1))
        else:
            # Extract steering vector from data
            steering_vector = data.get("steering_vector", None)
            if steering_vector is not None:
                steering_vectors_list.append(steering_vector.reshape(1, -1))

        # Stack all vectors for PCA projection
        all_mean_steering_vectors = np.vstack(mean_vectors + steering_vectors_list)
        all_mean_steering_vectors_reg = (
            all_mean_steering_vectors - all_vectors_mean
        ) / all_vectors_std

        # Apply PCA to get 2D projections
        try:
            pca = PCA(n_components=2, svd_solver="full")
            all_2d = pca.fit_transform(all_vectors_norm)
            mean_2d = pca.transform(all_mean_steering_vectors_reg)
        except (ValueError, RuntimeWarning) as e:
            print(
                f"Warning in {strategy}: PCA failed with error: {e}. Using simple dimension selection."
            )
            # Fallback: handle arrays with small dimensions correctly
            if all_vectors_norm.shape[1] <= 2:
                # If input has only 1 or 2 dimensions, use them directly
                if all_vectors_norm.shape[1] == 1:
                    # For 1D data, create a synthetic second dimension with small random values
                    all_2d = np.column_stack(
                        [
                            all_vectors_norm[:, 0],
                            np.random.normal(0, 0.01, size=all_vectors_norm.shape[0]),
                        ]
                    )
                    mean_2d = np.column_stack(
                        [
                            all_mean_steering_vectors_reg[:, 0],
                            np.random.normal(
                                0, 0.01, size=all_mean_steering_vectors_reg.shape[0]
                            ),
                        ]
                    )
                else:
                    # For 2D data, use both dimensions
                    all_2d = all_vectors_norm
                    mean_2d = all_mean_steering_vectors_reg
            else:
                # For higher dimensional data, select two dimensions with highest variance
                var = np.var(all_vectors_norm, axis=0)
                idx = np.argsort(var)[-2:]
                all_2d = all_vectors_norm[:, idx]
                mean_2d = all_mean_steering_vectors_reg[:, idx]

        # Split the projected points back to their categories
        start_idx = 0

        # Extract 2D coordinates for each category
        hate_yes_2d = all_2d[start_idx : start_idx + len(hate_yes_vectors)]
        start_idx += len(hate_yes_vectors)

        hate_no_2d = all_2d[start_idx : start_idx + len(hate_no_vectors)]
        start_idx += len(hate_no_vectors)

        safe_yes_2d = all_2d[start_idx : start_idx + len(safe_yes_vectors)]
        start_idx += len(safe_yes_vectors)

        safe_no_2d = all_2d[start_idx : start_idx + len(safe_no_vectors)]

        # Plot each category separately with consistent styling
        for category, points_2d in [
            ("hate_yes", hate_yes_2d),
            ("hate_no", hate_no_2d),
            ("safe_yes", safe_yes_2d),
            ("safe_no", safe_no_2d),
        ]:
            if len(points_2d) > 0:
                style = category_styles[category]
                ax.scatter(
                    points_2d[:, 0],
                    points_2d[:, 1],
                    color=style["color"],
                    alpha=0.6,
                    marker=style["marker"],
                    s=40,
                    edgecolors="black",
                    linewidths=0.5,
                    label=style["label"],
                    zorder=5,
                )

        # Plot mean vectors
        mean_idx = 0
        for category, mean_vector in [
            ("hate_yes", hate_yes_mean),
            ("hate_no", hate_no_mean),
            ("safe_yes", safe_yes_mean),
            ("safe_no", safe_no_mean),
        ]:
            if mean_vector is not None and mean_idx < len(mean_2d):
                style = category_styles[category]
                mean_2d_point = mean_2d[mean_idx]

                # Each category has its own label, ensuring each appears separately in the legend
                unique_label = f"{category} Mean Vector"

                ax.quiver(
                    0,
                    0,
                    mean_2d_point[0],
                    mean_2d_point[1],
                    color=style["color"],
                    label=unique_label,
                    scale_units="xy",
                    scale=1,  # Smaller scale makes arrows larger
                    width=0.02,  # Thicker arrow
                    headwidth=8,  # Larger head width
                    headlength=10,  # Larger head length
                    alpha=1.0,
                    zorder=20,  # Higher zorder to draw above other elements
                )
                mean_idx += 1

        # Plot steering vectors
        steering_idx = mean_idx
        if all_steering_vectors is not None:
            for name, data in all_steering_vectors.items():
                if steering_idx < len(mean_2d):
                    ax.quiver(
                        0,
                        0,
                        mean_2d[steering_idx][0],
                        mean_2d[steering_idx][1],
                        color=data["color"],
                        label=data["label"],
                        scale_units="xy",
                        scale=1,  # Smaller scale makes arrows larger
                        width=0.02,  # Thicker arrow
                        headwidth=8,  # Larger head width
                        headlength=10,  # Larger head length
                        alpha=1.0,
                        zorder=21,  # Higher zorder to draw above other elements
                    )
                    steering_idx += 1
        elif steering_vectors_list:
            # Plot default steering vector if specific ones aren't provided
            if steering_idx < len(mean_2d):
                ax.quiver(
                    0,
                    0,
                    mean_2d[steering_idx][0],
                    mean_2d[steering_idx][1],
                    color="#00FF00",  # Green for steering vector
                    label="Steering Vector",
                    scale_units="xy",
                    scale=1,  # Smaller scale makes arrows larger
                    width=0.02,  # Thicker arrow
                    headwidth=8,  # Larger head width
                    headlength=10,  # Larger head length
                    alpha=1.0,
                    zorder=21,  # Higher zorder to draw above other elements
                )

        # Set axis properties
        ax.set_title(f"{title} (Layer {layer_idx})")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Calculate axis limits from all data points
        all_points = np.vstack([hate_yes_2d, hate_no_2d, safe_yes_2d, safe_no_2d])
        if len(all_points) > 0:
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add a red border around the current strategy subplot
        if strategy == current_strategy:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    # Add explanation text
    plt.figtext(
        0.5,
        0.01,
        "These plots show the distribution of different statement types in 2D PCA space.\n"
        "Red circles: Hate statements with 'Yes', Blue circles: Hate statements with 'No'\n"
        "Blue triangles: Safe statements with 'Yes', Red triangles: Safe statements with 'No'\n"
        "Arrows show mean vectors and steering vectors.\n"
        "The red border indicates the embedding strategy used in the current run.",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.97))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()  # Close the plot to avoid memory issues
    return fig
